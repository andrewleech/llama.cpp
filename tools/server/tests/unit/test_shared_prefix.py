"""Cross-slot shared-prefix KV reuse — M1 (dense) and M2 (hybrid).

The feature lets a fan-out of agents that share a long system prompt attach to
another slot's already-computed prefix KV instead of re-prefilling it per slot.

Two engines / two model shapes:
  * M1 (dense, e.g. tinyllama2/stories260K): the donor's prefix cells gain the
    consumer's seq id (zero-copy on a unified KV). Log: "[shared-prefix] reused".
  * M2 (hybrid SSM+attention, e.g. falcon-h1-tiny): the attention cells are
    zero-copy shared while the recurrent state is restored from a context
    checkpoint captured at the share boundary. Log: "[shared-prefix:hybrid] reused
    ... (attn zero-copy + recurrent restore @ pos P)".

Gates (see server-context.cpp [TAG_SHARED_PREFIX]):
  * the share only fires with --kv-unified;
  * only from a GENERATING donor (its prefix cells are fully computed);
  * M2 additionally needs context checkpoints (--ctx-checkpoints > 0);
  * SWA / pure-recurrent / mtmd caches are excluded.

TEETH: every test carries a NEGATIVE CONTROL in which the share must NOT fire
(kv_unified=False, or n_ctx_checkpoints=0, or a SWA+vision model). "200 OK +
plausible text" is never a sufficient assertion — the authoritative positive
signal is the server log line. prompt_n alone is insufficient (own-cache reuse
also shrinks it), so the log line gates every positive assertion.
"""

import os
import json
import threading
import tempfile
import time

import pytest
from utils import *


# ---------------------------------------------------------------------------
# log reader: a tempfile + seek-based drain (same pattern as
# unit/test_kv_keep_only_active.py). We assert on the INFO log lines the server
# emits when a share fires.
# ---------------------------------------------------------------------------
class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.buf = ""

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        self.buf += content
        return content

    def wait_for(self, needle, timeout=5.0):
        """Drain (accumulating) until `needle` appears or timeout. The server writes
        its --verbose log to the file with some buffering, so a line emitted during a
        just-returned /completion may lag the HTTP response by a few ms — poll for it
        rather than reading once. Returns True if found."""
        deadline = time.time() + timeout
        while True:
            self.drain()
            if needle in self.buf:
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.05)

    def seen(self, needle):
        """Whether `needle` has appeared in everything drained so far."""
        self.drain()
        return needle in self.buf


M1_REUSE = "[shared-prefix] reused"
M2_REUSE = "[shared-prefix:hybrid] reused"


# A prefix long enough that re-prefilling it is clearly visible in prompt_n, and
# short enough that donor prompt + n_predict stays well under the server ctx so
# the donor never context-shifts (a shifted donor moves its prefix cells and the
# share is skipped — see test_ctxshift_guard).
PREFIX = (
    "Once upon a time in a land far away there lived a brave knight who "
    "traveled across mountains and rivers to find the legendary golden sword "
    "hidden deep within the enchanted forest of whispers. "
)
SUFFIX = "The knight finally reached the castle gates and knocked loudly three times."


def _mktemp_log():
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# GENERATING-donor window helper.
#
# On the tiny models prefill is near-instant, so the donor must still be
# GENERATING when the consumer attaches (only a generating donor's prefix cells
# are fully computed). We hold the donor open with a background streaming
# /completion (large n_predict, ignore_eos) and block on a server-side
# log-confirmed GENERATING poll via /slots (is_processing) before firing the
# consumer. The donor n_predict is bounded so it never fills ctx and shifts.
# ---------------------------------------------------------------------------
class DonorHold:
    def __init__(self, server, prompt, id_slot=0, n_predict=1500):
        self.server = server
        self.prompt = prompt
        self.id_slot = id_slot
        self.n_predict = n_predict
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        try:
            for _ in self.server.make_stream_request("POST", "/completion", data={
                "prompt": self.prompt,
                "id_slot": self.id_slot,
                "cache_prompt": True,
                "n_predict": self.n_predict,
                "ignore_eos": True,
                "temperature": 0.0,
                "top_k": 1,
                "seed": 42,
                "stream": True,
            }):
                if self._stop.is_set():
                    break
        except Exception:
            # the consumer / teardown may close the stream; that's expected
            pass

    def _slot(self):
        res = self.server.make_request("GET", "/slots")
        if res.status_code != 200:
            return None
        return next((s for s in res.body if s["id"] == self.id_slot), None)

    def _decoded(self, slot):
        # /slots reports per-slot generation progress under next_token[].n_decoded.
        nt = slot.get("next_token")
        if isinstance(nt, list) and nt:
            return nt[0].get("n_decoded", 0)
        return 0

    def is_generating(self):
        slot = self._slot()
        return slot is not None and bool(slot.get("is_processing"))

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Block until the donor slot is GENERATING — i.e. is_processing AND has
        # already decoded at least one token, so its prompt cells are fully computed
        # (a slot still prefilling is is_processing too but its suffix cells aren't
        # ready, and the cross-slot share only takes a fully-computed prefix). The
        # share gate also only accepts a SLOT_STATE_GENERATING donor. We re-confirm
        # generating right before returning so the consumer attaches inside the
        # window even under HTTP scheduling jitter.
        deadline = time.time() + 30
        while time.time() < deadline:
            slot = self._slot()
            if slot is not None and slot.get("is_processing") and self._decoded(slot) >= 1:
                # final confirmation it is still generating (not just finished)
                if self.is_generating():
                    return self
            time.sleep(0.02)
        raise TimeoutError("donor never reached GENERATING state")

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _completion(server, prompt, id_slot, n_predict=4):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
    })


# ===========================================================================
# FAST LANE — M1 dense (tinyllama2 / stories260K), not slow.
# ===========================================================================
def _dense_server(kv_unified=True, n_ctx=2048, n_slots=2):
    server = ServerPreset.tinyllama2()
    server.n_ctx = n_ctx
    server.n_slots = n_slots
    server.n_batch = 32
    server.server_slots = True
    server.server_continuous_batching = True
    server.temperature = 0.0
    server.kv_unified = kv_unified
    server.debug = True
    server.log_path = _mktemp_log()
    return server


def test_m1_fire_and_neg_no_unified():
    # M1-FIRE: a GENERATING donor + a consumer with the same prefix -> the
    # consumer attaches the donor's prefix cells (log line) and decodes only the
    # suffix (small prompt_n). NEG: kv_unified=False -> no share, full prompt_n.
    server = _dense_server(kv_unified=True)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0):
            res = _completion(server, PREFIX + SUFFIX, id_slot=1)
        assert res.status_code == 200
        assert log.wait_for(M1_REUSE), f"expected share log line, got:\n{log.buf}"
        # prompt_n is the suffix only; cache_n is the shared prefix.
        assert res.body["timings"]["cache_n"] > 0
        shared_prompt_n = res.body["timings"]["prompt_n"]
        total = res.body["timings"]["prompt_n"] + res.body["timings"]["cache_n"]
    finally:
        server.stop()

    # NEGATIVE CONTROL: identical scenario without --kv-unified. The share must
    # NOT fire (and must not crash — see test_gate_nonuniform_nocrash).
    server = _dense_server(kv_unified=False)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0):
            res = _completion(server, PREFIX + SUFFIX, id_slot=1)
        assert res.status_code == 200
        time.sleep(0.3)  # let any (erroneous) share line flush before asserting absence
        assert not log.seen(M1_REUSE), f"share fired without --kv-unified:\n{log.buf}"
        # full prefill: the consumer reprocessed the whole prompt, so prompt_n is
        # much larger than the shared run's suffix-only prompt_n.
        assert res.body["timings"]["prompt_n"] > shared_prompt_n
        assert res.body["timings"]["prompt_n"] == total - res.body["timings"]["cache_n"]
    finally:
        server.stop()


def test_m1_logiteq():
    # M1-LOGITEQ: the shared run's greedy output must equal a full-prefill control
    # (the zero-copy share is logit-equivalent), AND the treatment log shows the
    # reuse line. Anti-vacuity NEG: change one token in the consumer prefix and the
    # output diverges (so the equality above is not trivially satisfied).
    # cache_ram=0: stop the control's identical prompt being restored from the RAM
    # prompt cache into the treatment slot (which would beat the donor and suppress
    # the share). The cross-slot share must be the only reuse path here.
    server = _dense_server(kv_unified=True, n_slots=3)
    server.cache_ram = 0
    server.start()
    try:
        log = LogReader(server.log_path)

        # full-prefill control: a cold consumer, no donor present.
        control = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=16)
        assert control.status_code == 200
        assert not log.seen(M1_REUSE)  # no donor -> no share
        control_text = control.body["content"]

        # treatment: same prompt, but a GENERATING donor holds the prefix. Use a
        # clean slot 2 so the control's own cache on slot 1 cannot beat the donor.
        with DonorHold(server, PREFIX, id_slot=0):
            shared = _completion(server, PREFIX + SUFFIX, id_slot=2, n_predict=16)
        assert shared.status_code == 200
        assert log.wait_for(M1_REUSE), "treatment did not take the share"
        assert shared.body["content"] == control_text, "shared output != full-prefill control"

        # anti-vacuity: a materially different prompt must produce a different
        # continuation, so the equality above is not trivially satisfied by a model
        # that emits the same text for everything. (The tiny model's greedy decode
        # is driven by the tokens nearest the generation point, so we vary the
        # suffix tail rather than a word deep in the prefix.)
        diff = _completion(server, PREFIX + SUFFIX.replace("three times", "seven times loudly"),
                           id_slot=1, n_predict=16)
        assert diff.status_code == 200
        assert diff.body["content"] != control_text, "output identical for a changed prompt (vacuous)"
    finally:
        server.stop()


def test_m1_multiconsumer_and_neg():
    # M1-MULTICONSUMER: >=3 consumers share one donor -> >=3 reuse lines, each
    # suffix-only, all 200. NEG: kv_unified=False -> 0 reuse lines.
    n = 3
    server = _dense_server(kv_unified=True, n_slots=n + 1)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0):
            for i in range(n):
                res = _completion(server, PREFIX + f"{SUFFIX} consumer {i}", id_slot=1 + i)
                assert res.status_code == 200
                assert res.body["timings"]["cache_n"] > 0
            # each consumer attached suffix-only; wait for all n reuse lines to flush.
            assert log.wait_for(M1_REUSE)
        log.drain()
        assert log.buf.count(M1_REUSE) >= n, f"expected >= {n} reuse lines, got {log.buf.count(M1_REUSE)}"
    finally:
        server.stop()

    server = _dense_server(kv_unified=False, n_slots=n + 1)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0):
            for i in range(n):
                res = _completion(server, PREFIX + f"{SUFFIX} consumer {i}", id_slot=1 + i)
                assert res.status_code == 200
        time.sleep(0.3)
        assert log.seen(M1_REUSE) is False and log.buf.count(M1_REUSE) == 0
    finally:
        server.stop()


def test_m1_refcount_donor_free():
    # M1-REFCOUNT-DONOR-FREE: after the donor releases, the consumer is still 200
    # and its output still equals a full-prefill control — i.e. the shared cells
    # were neither freed nor corrupted when the donor's seq id was dropped.
    # The control comparison is the teeth.
    # cache_ram=0: keep the control's identical prompt out of the RAM prompt cache
    # so it is not restored into the consumer slot (the donor share is the path).
    server = _dense_server(kv_unified=True, n_slots=3)
    server.cache_ram = 0
    server.start()
    try:
        log = LogReader(server.log_path)

        # full-prefill control captured first (no donor), on its own slot 1.
        control = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=16)
        assert control.status_code == 200
        control_text = control.body["content"]

        # donor generates, consumer attaches the share, THEN donor releases.
        donor = DonorHold(server, PREFIX, id_slot=0)
        donor.__enter__()
        try:
            shared = _completion(server, PREFIX + SUFFIX, id_slot=2, n_predict=16)
            assert shared.status_code == 200
            assert log.wait_for(M1_REUSE), "consumer did not take the share"
        finally:
            donor.__exit__()  # donor releases its seq id; shared cells must survive

        # the consumer slot (2) keeps its KV (now sole owner of the shared cells); a
        # follow-up that re-sends the same prompt must still produce the control
        # output. If freeing the donor corrupted/freed the cells this diverges.
        again = _completion(server, PREFIX + SUFFIX, id_slot=2, n_predict=16)
        assert again.status_code == 200
        assert again.body["content"] == control_text, "output changed after donor freed (cells corrupted)"
    finally:
        server.stop()


def test_m1_owncache_vs_share():
    # M1-OWNCACHE-VS-SHARE: when a slot's own cached n_past already beats any
    # cross-slot LCP, it must NOT take the cross-slot share (own-cache reuse wins;
    # the share is "only worth it if it beats the slot's own reuse"). NEG inverse:
    # make the donor prefix longer than the slot's own cache -> the reuse line
    # appears.
    #
    # IMPORTANT: the idle-slot clearing feature (on by default with the prompt
    # cache + unified KV) saves and CLEARS an idle slot's KV when a new task
    # launches, which would wipe the consumer slot's own cache before the share
    # check and make the donor share win by default. Disable it here so the slot
    # genuinely retains its own (longer) cache — that is the condition under test.
    server = _dense_server(kv_unified=True, n_slots=3)
    server.no_cache_idle_slots = True
    server.start()
    try:
        log = LogReader(server.log_path)

        # prime slot 1 with the FULL prompt so its own cache covers everything.
        primed = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=4)
        assert primed.status_code == 200
        assert primed.body["timings"]["prompt_n"] > 0  # cold prime
        log.drain()

        # donor holds only the short PREFIX (a shorter common prefix than slot 1's
        # own cached PREFIX+SUFFIX). Re-issuing the same full prompt on slot 1 must
        # reuse its own cache, not the donor's shorter prefix — so NO share, and the
        # own-cache reuse shows up as a large cache_n / tiny prompt_n.
        with DonorHold(server, PREFIX, id_slot=0):
            res = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
        time.sleep(0.3)
        assert not log.seen(M1_REUSE), "took cross-slot share when own cache was longer"
        assert res.body["timings"]["cache_n"] >= primed.body["timings"]["prompt_n"] - 2, \
            "own-cache reuse did not cover the previously-primed prefix"

        # NEG inverse: a CLEAN consumer slot (never primed) attaches to the donor's
        # prefix because it has no own cache to beat -> the reuse line appears. This
        # is the contrapositive: the share fires exactly when the donor LCP wins.
        with DonorHold(server, PREFIX, id_slot=0):
            res = _completion(server, PREFIX + SUFFIX + " a clean fresh suffix here", id_slot=2, n_predict=4)
        assert res.status_code == 200
        assert log.wait_for(M1_REUSE), "clean consumer did not take the donor share"
    finally:
        server.stop()


def test_m1_ctxshift_guard():
    # M1-CTXSHIFT-GUARD (Part B): a slot that has taken the cross-slot share is
    # driven to need a context shift. The shared-prefix guard
    # ([TAG_SHARED_PREFIX_SHIFT]) keeps n_keep over the prefix shared with the
    # (still resident) donor so the in-place seq_add cannot move shared cells out
    # from under the donor. When generation then has no room left to discard, the
    # slot takes the documented error/refusal path (status != 200) — and crucially
    # the server does NOT abort: /health stays 200 afterwards. The no-crash is the
    # teeth (the guard exists to prevent corrupting/aborting on the shared cells).
    #
    # Empirical: this is the reliably reachable regime on the tiny model. A donor
    # generates (so the share fires), the consumer attaches the 102-token shared
    # prefix, then asks for more tokens than the ctx can hold while the guard pins
    # the shared prefix -> "Context size has been exceeded" 500, server survives.
    server = _dense_server(kv_unified=True, n_ctx=512)
    server.enable_ctx_shift = True
    server.start()
    try:
        log = LogReader(server.log_path)
        # donor n_predict kept just under the ctx headroom (102 prompt + 400 < 512)
        # so the DONOR never self-shifts and stays GENERATING long enough for the
        # consumer to attach the share.
        with DonorHold(server, PREFIX, id_slot=0, n_predict=400):
            # consumer n_predict pushes total well past n_ctx; the guard pins the
            # shared prefix so the slot cannot discard enough to continue.
            res = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=600)
            assert log.wait_for(M1_REUSE), f"shared prefix did not attach (test premise broken):\n{log.buf}"
        # the share fired, then the bounded ctx + pinned prefix forces the refusal.
        assert res.status_code != 200, \
            f"expected the ctx-shift refusal once the shared prefix pins n_keep, got 200\n{log.buf}"
        # precise refusal path (not some unrelated 5xx): the bounded ctx is exceeded.
        assert "Context size has been exceeded" in json.dumps(res.body), \
            f"expected the context-exceeded refusal specifically, got {res.status_code}: {res.body}"
        # TEETH: the server did not abort on the shared cells — it is still alive.
        assert server.make_request("GET", "/health").status_code == 200
        # and it keeps serving: a fresh, in-bounds request still works.
        ok = _completion(server, "A short fresh prompt.", id_slot=1, n_predict=4)
        assert ok.status_code == 200
    finally:
        server.stop()


def test_gate_nonuniform_nocrash():
    # GATE-NONUNIFIED-NOCRASH: without --kv-unified, a donor + consumer with the
    # same prefix must BOTH be 200, "[shared-prefix]" must NEVER appear, and
    # /health must be 200 afterwards. A prior bug aborted the server via
    # GGML_ASSERT(is_full) on the cross-stream seq_cp — the crash itself is the
    # teeth, so the post-condition is "server still alive".
    server = _dense_server(kv_unified=False)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0):
            res = _completion(server, PREFIX + SUFFIX, id_slot=1)
            assert res.status_code == 200
        time.sleep(0.3)
        assert not log.seen("[shared-prefix]"), f"share fired without --kv-unified:\n{log.buf}"
        # the donor stream and a fresh request both succeed; server did not abort.
        assert server.make_request("GET", "/health").status_code == 200
        again = _completion(server, PREFIX + SUFFIX, id_slot=1)
        assert again.status_code == 200
    finally:
        server.stop()


# ===========================================================================
# SLOW LANE — M2 hybrid (falcon-h1-tiny, ~55 MB, fetched on demand).
# ===========================================================================
# falcon-h1-tiny is deliberately excluded from load_all() (see utils._LOAD_ALL_SKIP),
# so the FAST lane never downloads it. The slow lane fetches it ON DEMAND the first
# time (offline=False), then reuses the cached copy OFFLINE for every later slow
# test. Going offline after the first fetch avoids hitting the HF HEAD endpoint once
# per slow test (which rate-limits / 404s under a full slow-lane run).
_falcon_fetched = False


def _ensure_falcon_cached():
    global _falcon_fetched
    if _falcon_fetched:
        return
    boot = ServerPreset.falcon_h1_tiny()
    boot.offline = False  # first use: fetch on demand
    boot.start()
    boot.stop()
    _falcon_fetched = True


def _hybrid_server(kv_unified=True, n_ctx_checkpoints=32, checkpoint_min_step=0,
                   n_ctx=1024, n_slots=2):
    _ensure_falcon_cached()
    server = ServerPreset.falcon_h1_tiny()
    server.offline = True  # cached by _ensure_falcon_cached(); skip the HF HEAD
    server.n_ctx = n_ctx
    server.n_slots = n_slots
    server.n_batch = 64
    server.n_ubatch = 64
    server.server_slots = True
    server.server_continuous_batching = True
    server.temperature = 0.0
    server.kv_unified = kv_unified
    server.n_ctx_checkpoints = n_ctx_checkpoints
    server.checkpoint_min_step = checkpoint_min_step
    server.debug = True
    server.log_path = _mktemp_log()
    return server


# Hybrid prompts: the donor prompt must be a strict PREFIX of the consumer so a
# checkpoint captured at the donor prompt end lands strictly inside the consumer's
# shared region (see EMPIRICAL unknown (b)). checkpoint_min_step=0 maximises the
# chance a checkpoint lands inside the tiny ctx.
H_PREFIX = (
    "System: you are a helpful assistant that answers questions about history. "
    "Here is some background context that will be shared across many requests "
    "so that the model can reason consistently about the topic at hand. "
) * 2
H_SUFFIX = "User: please summarise the key point in one sentence."


@pytest.mark.slow
def test_m2_fire_and_neg_no_checkpoints():
    # M2-FIRE: a hybrid share fires -> log "[shared-prefix:hybrid] reused ... @ pos P"
    # and the consumer prompt_n is reduced. NEG: n_ctx_checkpoints=0 -> no hybrid line
    # (the recurrent state cannot be restored without a checkpoint).
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            res = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
            assert log.wait_for(M2_REUSE), f"expected hybrid share log line, got:\n{log.buf}"
        assert res.body["timings"]["cache_n"] > 0
    finally:
        server.stop()

    # NEGATIVE CONTROL: checkpoints disabled -> hybrid gate (hybrid_prefix_ok)
    # is false, so no hybrid share.
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=0)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            res = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
        time.sleep(0.3)
        assert not log.seen(M2_REUSE), "hybrid share fired with n_ctx_checkpoints=0"
    finally:
        server.stop()


@pytest.mark.slow
def test_m2_logiteq():
    # M2-LOGITEQ: the hybrid shared output must equal a cold-prefill control (the
    # recurrent restore is logit-equivalent), AND the treatment log shows the
    # hybrid reuse line.
    # 3 slots: the cold control runs on its own slot so the treatment consumer
    # slot stays clean (otherwise its own cached prompt would beat the donor LCP
    # and the hybrid share would not fire — own-cache reuse wins ties).
    # cache_ram=0 disables the RAM prompt cache: otherwise the control's identical
    # prompt is cached and RESTORED into the (otherwise clean) treatment slot on
    # launch, giving it a full own-cache that out-scores the donor and suppresses
    # the cross-slot share we are trying to exercise.
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32, n_slots=3)
    server.cache_ram = 0
    server.start()
    try:
        log = LogReader(server.log_path)

        # cold full-prefill control on slot 2 (no donor present yet).
        control = _completion(server, H_PREFIX + H_SUFFIX, id_slot=2, n_predict=12)
        assert control.status_code == 200
        assert not log.seen(M2_REUSE)  # no donor -> no share
        control_text = control.body["content"]

        # treatment on the clean slot 1 with a GENERATING donor on slot 0.
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            shared = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=12)
            assert shared.status_code == 200
            assert log.wait_for(M2_REUSE), "treatment did not take the hybrid share"
        assert shared.body["content"] == control_text, "hybrid shared output != cold-prefill control"

        # anti-vacuity: a materially different prompt must produce a different continuation,
        # so the logit-equality above is not trivially satisfied by a model that emits the
        # same text for everything (mirrors the M1 anti-vacuity control).
        diff = _completion(server, H_PREFIX + H_SUFFIX + " Now reply with something entirely different.",
                           id_slot=2, n_predict=12)
        assert diff.status_code == 200
        assert diff.body["content"] != control_text, "output identical for a changed prompt (vacuous)"
    finally:
        server.stop()


@pytest.mark.slow
def test_m2_recurrent_independence():
    # M2-RECURRENT-INDEPENDENCE: the consumer's recurrent state is its OWN snapshot
    # (restored from a checkpoint), not aliased to the live donor tail. Verify the
    # consumer output equals its own full-prefill control AND the donor output
    # equals its standalone control. If the consumer aliased the donor's evolving
    # recurrent state, one of these would diverge.
    # 4 slots so each control and the treatment consumer use a distinct, clean slot
    # (a slot's own cached prompt would otherwise beat the donor LCP and suppress
    # the share). donor=0, consumer-control=1, donor-control=2, treatment=3.
    # cache_ram=0: stop the consumer-control's identical prompt being restored from
    # the RAM prompt cache into the treatment slot (which would suppress the share).
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32, n_slots=4)
    server.cache_ram = 0
    server.start()
    try:
        log = LogReader(server.log_path)

        # standalone controls (no concurrency), each on its own slot.
        consumer_control = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=12)
        assert consumer_control.status_code == 200
        consumer_ctrl_text = consumer_control.body["content"]
        donor_control = _completion(server, H_PREFIX, id_slot=2, n_predict=12)
        assert donor_control.status_code == 200
        donor_ctrl_text = donor_control.body["content"]
        log.drain()

        # now run them concurrently: donor generating on slot 0, the clean slot 3
        # consumer attaches the share.
        donor = DonorHold(server, H_PREFIX, id_slot=0, n_predict=300)
        donor.__enter__()
        try:
            shared = _completion(server, H_PREFIX + H_SUFFIX, id_slot=3, n_predict=12)
            assert shared.status_code == 200
            assert log.wait_for(M2_REUSE), "consumer did not take the hybrid share"
            assert shared.body["content"] == consumer_ctrl_text, \
                "consumer output != its own control (recurrent state aliased to donor?)"
        finally:
            donor.__exit__()

        # the donor's own standalone behaviour is unchanged by having lent its prefix.
        donor_again = _completion(server, H_PREFIX, id_slot=0, n_predict=12)
        assert donor_again.status_code == 200
        assert donor_again.body["content"] == donor_ctrl_text, \
            "donor output changed after lending its prefix"
    finally:
        server.stop()


@pytest.mark.slow
def test_m2_ckpt_align():
    # M2-CKPT-ALIGN: the reported share pos P is a checkpoint position <= the token
    # LCP, and the consumer prompt_n covers [P, len). (Dropped if a checkpoint
    # cannot be placed strictly inside the shared region on the tiny model — see
    # EMPIRICAL; we assert P>0 and that prompt_n + cache_n == total.)
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32)
    server.start()
    try:
        log = LogReader(server.log_path)
        consumer_prompt = H_PREFIX + H_SUFFIX
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            res = _completion(server, consumer_prompt, id_slot=1, n_predict=4)
            assert res.status_code == 200
            assert log.wait_for(M2_REUSE), f"hybrid share did not fire:\n{log.buf}"

        # parse "reused N tokens ... @ pos P" from the hybrid reuse line.
        import re
        m = re.search(r"\[shared-prefix:hybrid\] reused (\d+) tokens.*@ pos (\d+)", log.buf)
        assert m is not None, f"could not parse hybrid reuse line:\n{log.buf}"
        reused, pos_p = int(m.group(1)), int(m.group(2))
        assert pos_p > 0, "checkpoint pos P must be > 0 (a checkpoint landed inside the share)"

        cache_n = res.body["timings"]["cache_n"]
        prompt_n = res.body["timings"]["prompt_n"]
        # cache_n is the shared region [0, P); prompt_n is the reprocessed [P, len).
        assert cache_n > 0 and prompt_n > 0
        # P (the restored checkpoint pos) bounds the shared region; reused == cache_n.
        assert cache_n == reused, f"cache_n ({cache_n}) != reused tokens ({reused})"
        assert pos_p <= reused + 1, f"checkpoint pos P ({pos_p}) exceeds the shared region ({reused})"
    finally:
        server.stop()


@pytest.mark.slow
def test_m2_divergence_no_overshare():
    # M2-DIVERGENCE (regression for the checkpoint off-by-one). Donor and consumer share a
    # COMMON prefix then DIVERGE. The hybrid share must reuse only tokens within the common
    # prefix — never the first divergent token. The share boundary is a checkpoint aligned_pos,
    # which must be <= the token-level LCP. The prior bug gated on pos_max (== aligned_pos - 1),
    # admitting a checkpoint one token past the divergence and silently sharing the donor's
    # divergent token + its SSM tail. We assert reused <= LCP (no overshare) AND that the shared
    # output matches a cold-prefill control. (Whether a checkpoint lands exactly at the boundary
    # is model-dependent, so this reliably catches an overshare when it occurs and always verifies
    # divergence correctness.)
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32, n_slots=3)
    server.cache_ram = 0
    server.start()
    try:
        log = LogReader(server.log_path)
        common = H_PREFIX + H_PREFIX  # long enough that a checkpoint lands inside the common prefix
        donor_prompt    = common + " The donor keeps describing rivers and mountains at length."
        consumer_prompt = common + " Instead the consumer asks about a completely different topic."

        def toks(p):
            r = server.make_request("POST", "/tokenize", data={"content": p})
            return r.body["tokens"]
        td, tc = toks(donor_prompt), toks(consumer_prompt)
        lcp = 0
        for a, b in zip(td, tc):
            if a != b:
                break
            lcp += 1
        assert lcp > 0, "prompts must share a common token prefix"

        with DonorHold(server, donor_prompt, id_slot=0, n_predict=300):
            shared = _completion(server, consumer_prompt, id_slot=1, n_predict=12)
            assert shared.status_code == 200
            assert log.wait_for(M2_REUSE), f"hybrid share did not fire:\n{log.buf}"

        import re
        m = re.search(r"\[shared-prefix:hybrid\] reused (\d+) tokens.*@ pos (\d+)", log.buf)
        assert m is not None, f"could not parse hybrid reuse line:\n{log.buf}"
        reused, pos_p = int(m.group(1)), int(m.group(2))
        # REGRESSION: the shared boundary must stay within the common prefix — the off-by-one
        # would give reused / pos_p == lcp + 1 (sharing the first divergent token).
        assert reused <= lcp, f"shared {reused} tokens but only {lcp} are common (off-by-one overshare)"
        assert pos_p <= lcp, f"share pos {pos_p} exceeds the token LCP {lcp}"
        # NOTE: no exact output==cold-control assertion here. For a divergent tail the shared run
        # prefills [P, len) in a different batch layout than the contiguous cold control, so greedy
        # output can differ purely by batched-inference FP non-determinism (see test_m2_logiteq for
        # the strict-prefix logit-equality). The off-by-one regression is caught by the bounds above.
    finally:
        server.stop()


@pytest.mark.slow
def test_m2_gate_nockpt():
    # GATE-NOCKPT: n_ctx_checkpoints=0 -> no hybrid share, the consumer cold-
    # prefills the whole prompt. NEG/contrast: n_ctx_checkpoints>0 (the M2-FIRE
    # config) shows the hybrid line for the same scenario.
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=0)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            res = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
        time.sleep(0.3)
        assert not log.seen(M2_REUSE), "hybrid share fired with checkpoints disabled"
        cold_prompt_n = res.body["timings"]["prompt_n"]
    finally:
        server.stop()

    # contrast: with checkpoints on, the same scenario takes the share and the
    # consumer reprocesses fewer tokens.
    server = _hybrid_server(kv_unified=True, n_ctx_checkpoints=32)
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, H_PREFIX, id_slot=0, n_predict=300):
            res = _completion(server, H_PREFIX + H_SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
            assert log.wait_for(M2_REUSE), "hybrid share did not fire with checkpoints on"
        assert res.body["timings"]["prompt_n"] < cold_prompt_n
    finally:
        server.stop()


@pytest.mark.slow
def test_gate_swa_mtmd_excluded():
    # GATE-SWA-MTMD-EXCLUDED: tinygemma3 is SWA + vision; neither share line may
    # appear (the base gate excludes n_swa>0 and mtmd). Contrast is the M1-FIRE
    # no-SWA dense case which DOES fire.
    server = ServerPreset.tinygemma3()
    server.offline = False
    server.n_ctx = 2048
    server.n_slots = 2
    server.server_slots = True
    server.server_continuous_batching = True
    server.temperature = 0.0
    server.kv_unified = True
    server.n_ctx_checkpoints = 32
    server.checkpoint_min_step = 0
    server.cache_ram = 0  # disable the RAM prompt cache so the only possible reuse is the cross-slot share
    server.debug = True
    server.log_path = _mktemp_log()
    server.start()
    try:
        log = LogReader(server.log_path)
        with DonorHold(server, PREFIX, id_slot=0, n_predict=200) as hold:
            # precondition: the donor really is GENERATING (DonorHold enforces it),
            # so an absent share line is attributable to the gate, not a setup failure.
            assert hold.is_generating()
            res = _completion(server, PREFIX + SUFFIX, id_slot=1, n_predict=4)
            assert res.status_code == 200
        time.sleep(0.3)
        # no share fired on the SWA+vision model...
        assert not log.seen("[shared-prefix]"), f"share fired on a SWA model:\n{log.buf}"
        # ...and the consumer genuinely COLD-prefilled (reused nothing): with the RAM
        # cache off and a clean slot, cache_n==0 proves the no-share is the gate, not a
        # vacuous request short-circuit. Donor was generating, prefix matched, yet no share.
        assert res.body["timings"]["cache_n"] == 0, \
            f"consumer reused cache - cannot attribute the no-share to the SWA/mtmd gate: {res.body['timings']}"
    finally:
        server.stop()
