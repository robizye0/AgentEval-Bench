# AgentEval-Bench Bug-Finding & Fix Report

**Target:** `E:/AgentEval-Bench` (CLI/package, FastAPI telemetry, React dashboard)
**Date:** 2026-11-02
**Scope:** Exploratory testing and regression fixes for telemetry idempotency, rollback sandboxing, dashboard details, endpoint-field readability, and build/test health.
**Tester:** Hermes Agent (automated exploratory QA)

---

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 2 |
| Medium | 1 |
| Low | 1 |
| **Total** | **4** |

**Overall Assessment:** Four reproducible defects were fixed. Regression tests, the full test suite, lint/diff checks, package/dashboard builds, and an isolated browser/API check passed afterward.

---

## Issues

### Issue #1: Reused telemetry event IDs could silently accept changed payloads

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Category** | Functional |
| **URL** | `POST /eval/ingest` |
| **Status** | Fixed |

**Description:** The idempotency cache used `(case_id, event_id)` as its key but did not compare a retry's payload. Reusing an event ID with different usage/trajectory data could return the old result as if the new payload had been accepted. While adding payload validation, an initial fingerprint also included server-generated `ToolCall.timestamp` defaults, which made an otherwise identical HTTP retry look different. The final implementation fingerprints explicitly supplied fields: identical retries return the original response; changed payloads receive HTTP 409.

**Steps to Reproduce (before fix):**
1. POST an event with `case_id=retry-case`, `event_id=retry-case-0`.
2. POST the same event ID again but change `token_count`.
3. Observe that the changed event was treated as the earlier event rather than rejected.

**Expected Behavior:** An exact retry is idempotent; a different payload using an already-consumed event ID is rejected with 409.

**Actual Behavior:** Before the fix, payload differences were silently ignored. After the fix, the regression test confirms exact retry = 200 with the same response and changed payload = 409.

**Evidence:** `tests/test_eval_ingest.py::test_reused_event_id_with_different_payload_is_a_conflict` (passed).

### Issue #2: Concurrent async sandbox calls could leak mock state

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Category** | Functional |
| **URL** | `agenteval/mock/sandbox.py` |
| **Status** | Fixed |

**Description:** Concurrent sandboxed async calls sharing the same mutable state could overlap snapshots and rollbacks. One call could snapshot another call's transient mutation, leaving state behind after both completed and making tests order-dependent.

**Steps to Reproduce (before fix):**
1. Decorate two async functions with `@sandbox_state(state)` using the same list/dict.
2. Have each mutate the state and yield with `await asyncio.sleep(0)`.
3. Run both with `asyncio.gather` and inspect the shared state after completion.

**Expected Behavior:** Calls sharing a state serialize around snapshot/mutation/restore, leaving the original state unchanged.

**Actual Behavior:** Before the fix, overlapping rollback windows could leak a mutation. The sandbox now shares per-state synchronization across decorators and serializes async calls per event loop; synchronous contexts are protected by a shared reentrant lock.

**Evidence:** `tests/test_sandbox.py::test_sandbox_async_calls_serialize_shared_state` (passed using separate decorators).

### Issue #3: Selecting a dashboard trajectory did not reveal its result details

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Category** | Functional |
| **URL** | `http://127.0.0.1:5173` and `GET /eval/dashboard` |
| **Status** | Fixed |

**Description:** Clicking a trajectory toggled the row indicator, but no JSON detail panel appeared. The React UI rendered the panel only when a selected case contained `result`; the dashboard endpoint returned summary fields but omitted that `result` object.

**Steps to Reproduce (before fix):**
1. Connect the dashboard to an API with a completed evaluation.
2. Click the trajectory row.
3. Observe the indicator change while the result panel remains absent.

**Expected Behavior:** The selected case expands and shows its evaluation result and details.

**Actual Behavior:** The dashboard snapshot omitted `result`. The endpoint now includes the latest result, and the UI displays the selected evaluation JSON. Verified in-browser against an isolated API instance after posting a completed passing case.

**Evidence:** `tests/test_eval_ingest.py::test_remote_ingest_accumulates_steps_and_dashboard_exposes_final_status` (passed); browser snapshot displayed the expanded JSON.

### Issue #4: API endpoint text was clipped in the dashboard connection control

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Category** | Visual / UX |
| **URL** | `http://127.0.0.1:5173` |
| **Status** | Fixed |

**Description:** At desktop width, the 360px connection control left too little room for the API URL beside the token and Connect controls. The final characters were clipped, making it difficult to visually verify the entered endpoint.

**Steps to Reproduce (before fix):**
1. Open the dashboard at desktop width.
2. Enter `http://127.0.0.1:8017`.
3. Observe that the final URL character is hidden in the endpoint field.

**Expected Behavior:** The endpoint is readable in the connection form.

**Actual Behavior:** The endpoint field was clipped. The control width is now 440px with a responsive max-width; the full endpoint is visible and the overall desktop layout remains intact.

**Evidence:** Verified visually in the browser after the CSS change. The browser image tool did not provide a local screenshot path for this report.

---

## Issues Summary Table

| # | Title | Severity | Category | Status |
|---|-------|----------|----------|--------|
| 1 | Reused telemetry event IDs could silently accept changed payloads | High | Functional | Fixed |
| 2 | Concurrent async sandbox calls could leak mock state | High | Functional | Fixed |
| 3 | Selecting a dashboard trajectory did not reveal its result details | Medium | Functional | Fixed |
| 4 | API endpoint text was clipped in the dashboard connection control | Low | Visual / UX | Fixed |

## Testing Coverage

### Pages and components tested
- React dashboard at `http://127.0.0.1:5173` connected to local telemetry API.
- `/eval/ingest`, `/eval/dashboard`, and SSE snapshot behavior.
- Sandbox rollback with concurrent async decorators.
- Installed console entry point help and debug-command help.

### Verification after fixes
- `python -m pytest -q` — **58 passed**.
- `python -m ruff check .` — passed.
- `git diff --check` — passed.
- `npm run build` — passed.
- `python -m build` — sdist and wheel built successfully.
- Browser/API smoke — completed event displayed as passed; expanding the row showed its JSON result; browser console had **0 errors**.

### Not tested / Out of scope
- PyPI publication and account/trusted-publisher setup.
- Multi-worker or production persistence/load behavior; telemetry remains in-memory.
- Live external LLM judge providers and real third-party API integrations.

### Blockers
- None for the tested local scope.

## Notes

- Pytest emitted one non-failing Starlette deprecation warning: its `TestClient` integration currently uses `httpx`, and the installed Starlette version recommends `httpx2` for future compatibility.
- Fixes are present in the local working tree and have not been committed or pushed.
