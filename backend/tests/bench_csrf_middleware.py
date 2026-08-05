"""
Benchmarking Harness — SecurityMiddleware (csrf.py)
=====================================================

Seven suites measuring five originally planned + two new dimensions:

  Suite 1  TestLatencyOverhead        — p50 / p99 / max per scenario (200 iters)
  Suite 2  TestThroughputCapacity     — RPS at concurrency 1/4/8/16 (3 s window)
  Suite 3  TestDetectionAccuracy      — FP / FN rates over 100 samples each
  Suite 4  TestResourceConsumption    — RSS memory delta + GC pressure (500-req burst)
  Suite 5  TestFailSafeBehavior       — mocked exceptions / slow-path fallback
  Suite 6  TestPayloadSizeSensitivity — latency vs body size 0B→2MB (req + res)
  Suite 7  TestSQLiteWALContention    — RPS + 5xx rate under concurrent SQLite writes

Run (opt-in via the `bench` mark):
    cd backend && pytest tests/bench_csrf_middleware.py -v -s -m bench

All Ollama / RAG dependencies are irrelevant — the middleware is tested in
isolation using a FastAPI TestClient with a temp SQLite DB.
"""

from __future__ import annotations

import gc
import json
import statistics
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar
from unittest.mock import patch

import httpx
import psutil
import pytest
import services.db_service as db
from app import app
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middleware.csrf import SecurityMiddleware, compute_request_hash
from starlette.responses import Response as StarletteResponse

# ── pytest mark registration ──────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "bench: performance / benchmarking tests")


# ── Shared temp DB (same pattern as test_csrf.py) ────────────────────────────

_tmp_db = tempfile.mktemp(suffix="_bench_csrf.db")
db.DB_PATH = _tmp_db
db.init_db()

client = TestClient(app, raise_server_exceptions=True)

_TESTCLIENT_HOST = "testclient"

VALID_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
]

ATTACKER_ORIGINS = [
    "https://evil.com",
    "http://attacker.local",
    "https://csrf-demo.io",
    "null",
]

# ── Utility helpers ───────────────────────────────────────────────────────────


def _compute_hash(method: str, path: str, body_dict: dict) -> str:
    """Mirror of the helper in test_csrf.py — kept local so this file is self-contained."""
    req = httpx.Request(method, f"http://testserver{path}", json=body_dict)
    return compute_request_hash(_TESTCLIENT_HOST, method, path, req.content)


def _percentile(data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (0-100) of *data*."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def _stats(latencies_ms: list[float]) -> dict[str, float]:
    """Return a dict of min / p50 / p99 / max / mean / stdev."""
    return {
        "n": len(latencies_ms),
        "min_ms": round(min(latencies_ms), 3),
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p99_ms": round(_percentile(latencies_ms, 99), 3),
        "max_ms": round(max(latencies_ms), 3),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "stdev_ms": round(statistics.stdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0,
    }


def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple aligned table to stdout (visible with -s)."""
    col_width = max(20, *(len(c) for c in columns))
    sep = "=" * (col_width * len(columns) + len(columns) + 1)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    header = " ".join(c.ljust(col_width) for c in columns)
    print(f"  {header}")
    print("-" * len(sep))
    for row in rows:
        line = " ".join(str(row.get(c, "")).ljust(col_width) for c in columns)
        print(f"  {line}")
    print(sep)


# ==============================================================================
# Suite 1 — Latency Overhead
# ==============================================================================


@pytest.mark.bench
class TestLatencyOverhead:
    """Measures p50, p99, and max response-time inflation per middleware scenario.

    Each scenario isolates a distinct code path through SecurityMiddleware.dispatch:
      - safe_method_get:       method gate short-circuits immediately (no body read)
      - clean_post_no_origin:  full path — hash, SQLite R/W, handler, cache write
      - valid_origin_post:     allowlisted origin — same as clean_post but with header parse
      - rejected_origin_403:   early exit at origin check — no SQLite writes
      - deduplicated_hit:      sentinel planted manually — cache read path only
    """

    ITERATIONS = 50
    P99_BUDGET_MS = 500.0

    def _run_scenario(self, fn, n: int = ITERATIONS) -> dict[str, float]:
        latencies: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - t0) * 1000)
        return _stats(latencies)

    def test_safe_method_get(self):
        stats = self._run_scenario(lambda: client.get("/api/sessions/"))
        _print_table(
            "Latency — safe_method_get",
            [stats],
            ["n", "min_ms", "p50_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms"],
        )
        assert stats["p99_ms"] < self.P99_BUDGET_MS, (
            f"p99 {stats['p99_ms']} ms exceeds budget {self.P99_BUDGET_MS} ms"
        )

    def test_clean_post_no_origin(self):
        # Each call uses a unique title to avoid deduplication cache hits.
        stats = self._run_scenario(
            lambda: client.post("/api/sessions/", json={"title": f"bench-{uuid.uuid4().hex}"})
        )
        _print_table(
            "Latency — clean_post_no_origin",
            [stats],
            ["n", "min_ms", "p50_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms"],
        )
        assert stats["p99_ms"] < self.P99_BUDGET_MS

    def test_valid_origin_post(self):
        stats = self._run_scenario(
            lambda: client.post(
                "/api/sessions/",
                json={"title": f"bench-valid-{uuid.uuid4().hex}"},
                headers={"Origin": "http://localhost:3000"},
            )
        )
        _print_table(
            "Latency — valid_origin_post",
            [stats],
            ["n", "min_ms", "p50_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms"],
        )
        assert stats["p99_ms"] < self.P99_BUDGET_MS

    def test_rejected_origin_403(self):
        stats = self._run_scenario(
            lambda: client.post(
                "/api/sessions/",
                json={"title": "will-be-blocked"},
                headers={"Origin": "https://evil.com"},
            )
        )
        _print_table(
            "Latency — rejected_origin_403",
            [stats],
            ["n", "min_ms", "p50_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms"],
        )
        assert stats["p99_ms"] < self.P99_BUDGET_MS

    def test_deduplicated_cache_hit(self):
        """Plant a 'done' sentinel and measure latency of serving from cache."""
        body = {"title": "dedupe-bench-hit"}
        req_hash = _compute_hash("POST", "/api/sessions/", body)
        fake_resp = json.dumps({"id": "cached-id", "title": "dedupe-bench-hit"}).encode()
        db.dedupe_set_done(
            req_hash,
            response_body=fake_resp,
            status_code=200,
            headers={"content-type": "application/json"},
            expires_at=time.time() + 300,
        )

        stats = self._run_scenario(
            lambda: client.post("/api/sessions/", json=body)
        )
        _print_table(
            "Latency — deduplicated_cache_hit",
            [stats],
            ["n", "min_ms", "p50_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms"],
        )
        assert stats["p99_ms"] < self.P99_BUDGET_MS

        # Cleanup
        db.dedupe_delete(req_hash)


# ==============================================================================
# Suite 2 — Throughput Capacity
# ==============================================================================


@pytest.mark.bench
class TestThroughputCapacity:
    """Measures requests per second (RPS) at increasing concurrency levels.

    Uses ThreadPoolExecutor so multiple in-flight requests compete for the
    TestClient, revealing degradation under parallel load.
    """

    WINDOW_SECONDS = 3.0
    CONCURRENCY_LEVELS: ClassVar[list[int]] = [1, 4, 8, 16]
    BASELINE_RPS_FLOOR = 10.0  # must achieve at least 10 RPS at concurrency=1

    def _run_rps(self, concurrency: int) -> float:
        """Return RPS achieved over WINDOW_SECONDS with *concurrency* threads."""
        results: list[int] = []
        stop_event = threading.Event()

        def worker():
            count = 0
            while not stop_event.is_set():
                r = client.post(
                    "/api/sessions/",
                    json={"title": f"tput-{uuid.uuid4().hex}"},
                )
                if r.status_code in (200, 409):
                    count += 1
            results.append(count)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(worker) for _ in range(concurrency)]
            time.sleep(self.WINDOW_SECONDS)
            stop_event.set()
            for f in as_completed(futs):
                f.result()  # surface any thread exception

        total_requests = sum(results)
        return round(total_requests / self.WINDOW_SECONDS, 2)

    def test_throughput_all_concurrency_levels(self):
        rows = []
        baseline_rps = None

        for c in self.CONCURRENCY_LEVELS:
            rps = self._run_rps(c)
            if baseline_rps is None:
                baseline_rps = rps
            degradation = round(rps / baseline_rps * 100, 1) if baseline_rps else 100.0
            rows.append({"concurrency": c, "rps": rps, "vs_baseline_%": degradation})

        _print_table(
            "Throughput Capacity — RPS vs Concurrency",
            rows,
            ["concurrency", "rps", "vs_baseline_%"],
        )

        assert baseline_rps is not None and baseline_rps >= self.BASELINE_RPS_FLOOR, (
            f"Baseline RPS {baseline_rps} is below floor {self.BASELINE_RPS_FLOOR}"
        )


# ==============================================================================
# Suite 3 — Detection Accuracy
# ==============================================================================


@pytest.mark.bench
class TestDetectionAccuracy:
    """Assesses false-positive and false-negative rates over statistical samples."""

    SAMPLES = 100

    def test_false_positive_rate_clean_origins(self):
        """No clean request (valid or no origin) should ever be rejected."""
        blocked = 0
        origins = [None, *VALID_ORIGINS]
        for i in range(self.SAMPLES):
            origin = origins[i % len(origins)]
            headers = {"Origin": origin} if origin else {}
            r = client.post(
                "/api/sessions/",
                json={"title": f"fp-test-{uuid.uuid4().hex}"},
                headers=headers,
            )
            if r.status_code == 403:
                blocked += 1

        fp_rate = blocked / self.SAMPLES * 100
        print(f"\n  False-positive rate (clean origins): {fp_rate:.1f}%  ({blocked}/{self.SAMPLES} blocked)")
        assert blocked == 0, f"False positives detected: {blocked}/{self.SAMPLES}"

    def test_false_negative_rate_attacker_origins(self):
        """Every attacker-origin request must be blocked."""
        passed = 0
        for i in range(self.SAMPLES):
            origin = ATTACKER_ORIGINS[i % len(ATTACKER_ORIGINS)]
            r = client.post(
                "/api/sessions/",
                json={"title": "should-be-blocked"},
                headers={"Origin": origin},
            )
            if r.status_code != 403:
                passed += 1

        fn_rate = passed / self.SAMPLES * 100
        print(f"\n  False-negative rate (attacker origins): {fn_rate:.1f}%  ({passed}/{self.SAMPLES} passed)")
        assert passed == 0, f"False negatives detected: {passed}/{self.SAMPLES}"

    def test_referer_false_positive_rate(self):
        """Valid Referer headers must not trigger false 403s."""
        valid_referers = [
            "http://localhost:3000/chat",
            "http://127.0.0.1:3000/",
            "http://localhost:5173/settings",
        ]
        blocked = 0
        for i in range(50):
            referer = valid_referers[i % len(valid_referers)]
            r = client.post(
                "/api/sessions/",
                json={"title": f"referer-fp-{uuid.uuid4().hex}"},
                headers={"Referer": referer},
            )
            if r.status_code == 403:
                blocked += 1

        print(f"\n  Referer false-positive rate: {blocked}/50 blocked")
        assert blocked == 0

    def test_referer_false_negative_rate(self):
        """Attacker Referer headers (no Origin) must all be blocked."""
        attacker_referers = [
            "https://evil.com/csrf",
            "http://attacker.local/form",
        ]
        passed = 0
        for i in range(50):
            referer = attacker_referers[i % len(attacker_referers)]
            r = client.post(
                "/api/sessions/",
                json={"title": "referer-fn-test"},
                headers={"Referer": referer},
            )
            if r.status_code != 403:
                passed += 1

        print(f"\n  Referer false-negative rate: {passed}/50 passed through")
        assert passed == 0


# ==============================================================================
# Suite 4 — Resource Consumption
# ==============================================================================


@pytest.mark.bench
class TestResourceConsumption:
    """Monitors RSS memory, CPU time, and GC pressure during a 500-request burst."""

    BURST = 500
    RSS_DELTA_LIMIT_MB = 50.0

    def test_memory_and_gc_during_burst(self):
        proc = psutil.Process()
        gc.collect()
        gc_before = gc.get_count()
        mem_before_mb = proc.memory_info().rss / 1024 / 1024
        cpu_before = proc.cpu_times()

        for _ in range(self.BURST):
            client.post("/api/sessions/", json={"title": f"burst-{uuid.uuid4().hex}"})

        gc.collect()
        gc_after = gc.get_count()
        mem_after_mb = proc.memory_info().rss / 1024 / 1024
        cpu_after = proc.cpu_times()

        rss_delta = round(mem_after_mb - mem_before_mb, 2)
        cpu_user_delta = round(cpu_after.user - cpu_before.user, 3)
        cpu_sys_delta = round(cpu_after.system - cpu_before.system, 3)
        gc_gen0_delta = gc_after[0] - gc_before[0]
        gc_gen1_delta = gc_after[1] - gc_before[1]

        print(
            f"\n  Resource Consumption ({self.BURST} requests):\n"
            f"    RSS delta:          {rss_delta:+.2f} MB\n"
            f"    CPU user time:      {cpu_user_delta:.3f} s\n"
            f"    CPU sys time:       {cpu_sys_delta:.3f} s\n"
            f"    GC gen0 delta:      {gc_gen0_delta}\n"
            f"    GC gen1 delta:      {gc_gen1_delta}"
        )
        assert rss_delta < self.RSS_DELTA_LIMIT_MB, (
            f"Memory grew by {rss_delta} MB — exceeds limit of {self.RSS_DELTA_LIMIT_MB} MB"
        )


# ==============================================================================
# Suite 5 — Fail-Safe Behavior
# ==============================================================================


@pytest.mark.bench
class TestFailSafeBehavior:
    """Tests middleware fallback when internal exceptions or slow paths are injected."""

    def test_origin_header_raises_runtime_error(self):
        """RuntimeError in _origin_from_header must return 500, not crash the server."""
        with patch(
            "middleware.csrf._origin_from_header",
            side_effect=RuntimeError("injected fault"),
        ):
            r = client.post("/api/sessions/", json={"title": "failsafe-runtime"})
        assert r.status_code == 500
        assert "Security verification error" in r.json().get("detail", "")

    def test_origin_header_raises_value_error(self):
        """ValueError must also trigger the 500 fail-closed path."""
        with patch(
            "middleware.csrf._origin_from_header",
            side_effect=ValueError("bad header encoding"),
        ):
            r = client.post("/api/sessions/", json={"title": "failsafe-value-error"})
        assert r.status_code == 500
        assert "Security verification error" in r.json().get("detail", "")

    def test_safe_methods_bypass_exception_path(self):
        """GET never enters the origin-check block — must succeed even with a patched raiser."""
        with patch(
            "middleware.csrf._origin_from_header",
            side_effect=RuntimeError("injected fault"),
        ):
            r = client.get("/api/sessions/")
        assert r.status_code == 200

    def test_slow_origin_check_latency_inflation(self):
        """A 20 ms simulated origin-check delay is measurable and the request still completes."""
        ARTIFICIAL_DELAY_S = 0.02
        import middleware.csrf as _csrf_mod

        real_fn = _csrf_mod._origin_from_header

        def slow_origin(request):
            time.sleep(ARTIFICIAL_DELAY_S)
            return real_fn(request)

        latencies: list[float] = []
        with patch("middleware.csrf._origin_from_header", side_effect=slow_origin):
            for _ in range(20):
                t0 = time.perf_counter()
                r = client.post(
                    "/api/sessions/",
                    json={"title": f"slow-{uuid.uuid4().hex}"},
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200

        s = _stats(latencies)
        print(
            f"\n  Slow-origin latency ({ARTIFICIAL_DELAY_S * 1000:.0f} ms injected):\n"
            f"    p50={s['p50_ms']} ms  p99={s['p99_ms']} ms  max={s['max_ms']} ms"
        )
        # p50 must at least reflect the injected delay (within 20% tolerance).
        assert s["p50_ms"] >= ARTIFICIAL_DELAY_S * 1000 * 0.8, (
            f"p50 {s['p50_ms']} ms is suspiciously low — mock may not have applied"
        )


# ==============================================================================
# Suite 6 — Payload Size Sensitivity  [NEW]
# ==============================================================================

# Build a minimal FastAPI app that wraps SecurityMiddleware around a single
# synthetic echo route.  This lets us vary response body size independently of
# the real application routes, and avoids polluting the sessions DB.
_payload_app = FastAPI()
_payload_app.add_middleware(
    SecurityMiddleware,
    allowed_origins=["http://localhost:3000"],
)


@_payload_app.post("/echo")
async def _echo_route(size: int = 0) -> StarletteResponse:
    """Return a response body of exactly *size* bytes."""
    return StarletteResponse(content=b"x" * size, status_code=200)


_payload_client = TestClient(_payload_app, raise_server_exceptions=True)

# Body sizes: 0 B, 1 KB, 64 KB, 512 KB, 2 MB
_PAYLOAD_SIZES = [
    ("0 B",    0),
    ("1 KB",   1_024),
    ("64 KB",  64 * 1_024),
    ("512 KB", 512 * 1_024),
    ("2 MB",   2 * 1_024 * 1_024),
]

P99_SOFT_CEILING_MS = 500.0  # generous soft ceiling — only fails on severely slow machines


@pytest.mark.bench
class TestPayloadSizeSensitivity:
    """Measures how dual body-buffering scales with payload size.

    The middleware performs two full-body copies on every non-GET request:
      1. ``req_body = await request.body()``  — for SHA-256 hash + dedupe key
      2. ``b"".join([chunk async for chunk in response.body_iterator])``  — to cache response

    Both copies are O(N) in body size. This suite quantifies the cost from
    typical API JSON (<=1 KB) through document-upload territory (2 MB).
    """

    ITERATIONS = 50

    # ── Request body scaling ──────────────────────────────────────────────────

    def test_request_body_size_scaling(self):
        """Latency vs increasing request body size (response is always tiny)."""
        rows = []
        for label, size in _PAYLOAD_SIZES:
            latencies: list[float] = []
            for _ in range(self.ITERATIONS):
                # Ensure unique content so we benchmark the full processing path each iter
                payload = (uuid.uuid4().hex.encode() + b"x" * size)[:size]
                t0 = time.perf_counter()
                r = _payload_client.post(
                    "/echo?size=0",
                    content=payload,
                    headers={
                        "content-type": "application/octet-stream",
                        "Origin": "http://localhost:3000",
                    },
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200, (
                    f"Unexpected {r.status_code} at request body size {label}"
                )
            s = _stats(latencies)
            rows.append({"body_size": label, "p50_ms": s["p50_ms"], "p99_ms": s["p99_ms"], "max_ms": s["max_ms"]})

            assert s["p99_ms"] < P99_SOFT_CEILING_MS, (
                f"Request body {label}: p99 {s['p99_ms']} ms exceeds ceiling {P99_SOFT_CEILING_MS} ms"
            )

        _print_table(
            "Payload Sensitivity — Request Body Size vs Latency",
            rows,
            ["body_size", "p50_ms", "p99_ms", "max_ms"],
        )

    # ── Response body scaling ─────────────────────────────────────────────────

    def test_response_body_size_scaling(self):
        """Latency vs increasing response body size (request body is always tiny).

        The middleware accumulates the full response body via b"".join before it
        can reconstruct the Response object. This test isolates that cost.
        """
        rows = []
        for label, size in _PAYLOAD_SIZES:
            latencies: list[float] = []
            for _ in range(self.ITERATIONS):
                t0 = time.perf_counter()
                r = _payload_client.post(
                    f"/echo?size={size}",
                    json={"probe": f"response-size-{size}-{uuid.uuid4().hex}"},
                    headers={"Origin": "http://localhost:3000"},
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200
                assert len(r.content) == size, (
                    f"Response body truncated: expected {size} B, got {len(r.content)} B"
                )
            s = _stats(latencies)
            rows.append({"response_size": label, "p50_ms": s["p50_ms"], "p99_ms": s["p99_ms"], "max_ms": s["max_ms"]})

            assert s["p99_ms"] < P99_SOFT_CEILING_MS, (
                f"Response body {label}: p99 {s['p99_ms']} ms exceeds ceiling {P99_SOFT_CEILING_MS} ms"
            )

        _print_table(
            "Payload Sensitivity — Response Body Size vs Latency",
            rows,
            ["response_size", "p50_ms", "p99_ms", "max_ms"],
        )

    # ── SHA-256 hashing overhead in isolation ─────────────────────────────────

    def test_sha256_hashing_overhead_isolation(self):
        """Unit-level benchmark of compute_request_hash() vs body size.

        Isolates the hashing cost from the HTTP roundtrip so both contributions
        can be compared independently.
        """
        rows = []
        for label, size in _PAYLOAD_SIZES:
            body_bytes = b"x" * size
            latencies: list[float] = []
            for _ in range(self.ITERATIONS):
                t0 = time.perf_counter()
                compute_request_hash("127.0.0.1", "POST", "/echo", body_bytes)
                latencies.append((time.perf_counter() - t0) * 1000)
            s = _stats(latencies)
            rows.append({"body_size": label, "p50_ms": s["p50_ms"], "p99_ms": s["p99_ms"], "max_ms": s["max_ms"]})

        _print_table(
            "SHA-256 Hashing Overhead (no HTTP) vs Body Size",
            rows,
            ["body_size", "p50_ms", "p99_ms", "max_ms"],
        )
        # Hashing alone should never exceed 50 ms even for 2 MB.
        worst = max(row["p99_ms"] for row in rows)
        assert worst < 50.0, (
            f"SHA-256 p99 {worst} ms exceeds 50 ms — investigate compute_request_hash()"
        )


# ==============================================================================
# Suite 7 — SQLite WAL Write Contention  [NEW]
# ==============================================================================


@pytest.mark.bench
class TestSQLiteWALContention:
    """Measures throughput degradation and server resilience under concurrent
    SQLite writes from the deduplication subsystem.

    Every non-safe request through SecurityMiddleware triggers up to 4 SQLite
    operations against the same WAL file:
      dedupe_purge_expired -> dedupe_get -> dedupe_set_processing -> dedupe_set_done

    ThreadPoolExecutor workers each hold their own TestClient instance and send
    unique POST requests over a fixed time window (no dedup cache hits — every
    request exercises the full SQLite write path).

    Metrics:
      - RPS at concurrency 1 / 10 / 20
      - 5xx error count (must be zero — lock retries must resolve silently)
      - RPS degradation ratio vs single-worker baseline
    """

    WINDOW_SECONDS = 5.0
    CONCURRENCY_LEVELS: ClassVar[list[int]] = [1, 10, 20]

    def _worker_fn(self, stop_event: threading.Event, worker_client: TestClient) -> dict:
        """Thread worker: send unique POSTs until stop_event is set."""
        ok = 0
        errors_5xx = 0
        errors_other = 0
        while not stop_event.is_set():
            try:
                r = worker_client.post(
                    "/api/sessions/",
                    json={"title": f"wal-bench-{uuid.uuid4().hex}"},
                )
                if r.status_code == 200:
                    ok += 1
                elif r.status_code >= 500:
                    errors_5xx += 1
                else:
                    errors_other += 1
            except Exception:  # noqa: BLE001
                errors_5xx += 1
        return {"ok": ok, "errors_5xx": errors_5xx, "errors_other": errors_other}

    def _run_contention_level(self, concurrency: int) -> dict:
        stop_event = threading.Event()
        futures_results: list[dict] = []

        # Each thread gets its own client to avoid sharing connection state.
        worker_clients = [
            TestClient(app, raise_server_exceptions=False)
            for _ in range(concurrency)
        ]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [
                pool.submit(self._worker_fn, stop_event, worker_clients[i])
                for i in range(concurrency)
            ]
            time.sleep(self.WINDOW_SECONDS)
            stop_event.set()
            for f in as_completed(futs):
                futures_results.append(f.result())

        total_ok = sum(r["ok"] for r in futures_results)
        total_5xx = sum(r["errors_5xx"] for r in futures_results)
        total_other = sum(r["errors_other"] for r in futures_results)
        rps = round(total_ok / self.WINDOW_SECONDS, 2)

        return {
            "concurrency": concurrency,
            "ok": total_ok,
            "errors_5xx": total_5xx,
            "errors_other": total_other,
            "rps": rps,
        }

    def test_wal_contention_all_levels(self):
        """Run all concurrency levels and assert zero 5xx errors at each."""
        rows = []
        baseline_rps: float | None = None

        for c in self.CONCURRENCY_LEVELS:
            result = self._run_contention_level(c)

            if baseline_rps is None:
                baseline_rps = result["rps"] or 1.0

            degradation_pct = round(result["rps"] / baseline_rps * 100, 1)
            result["vs_baseline_%"] = degradation_pct
            rows.append(result)

            # Core assertion: zero 5xx — lock retries must resolve within SQLite timeout.
            assert result["errors_5xx"] == 0, (
                f"Concurrency={c}: {result['errors_5xx']} 5xx errors detected — "
                "SQLite lock retries failed to resolve within timeout"
            )

        _print_table(
            f"SQLite WAL Contention — RPS & Errors ({self.WINDOW_SECONDS:.0f}s window per level)",
            rows,
            ["concurrency", "rps", "vs_baseline_%", "ok", "errors_5xx", "errors_other"],
        )

    def test_wal_no_data_corruption_under_concurrent_writes(self):
        """Verify concurrent writes never corrupt the dedupe_cache table.

        Spins up 50 threads simultaneously writing unique sentinels, then reads
        them all back and confirms every hash is present with the correct status.
        """
        NUM_WRITERS = 50
        hashes_written: list[str] = []
        lock = threading.Lock()

        def write_sentinel(_):
            h = uuid.uuid4().hex
            db.dedupe_set_processing(h, expires_at=time.time() + 60)
            with lock:
                hashes_written.append(h)

        with ThreadPoolExecutor(max_workers=NUM_WRITERS) as pool:
            list(pool.map(write_sentinel, range(NUM_WRITERS)))

        now = time.time()
        missing = [h for h in hashes_written if db.dedupe_get(h, now) is None]
        print(
            f"\n  WAL corruption check: {NUM_WRITERS} concurrent writes, "
            f"{len(missing)} missing after readback"
        )
        assert not missing, (
            f"{len(missing)}/{NUM_WRITERS} sentinels not found after concurrent writes: "
            f"{missing[:5]}"
        )

        # Cleanup
        for h in hashes_written:
            db.dedupe_delete(h)


# ==============================================================================
# Entry point for direct execution
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "bench"])
