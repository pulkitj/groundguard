# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Project Is

**agentic-verifier** is a Python middleware library (MIT, LLM-agnostic) that verifies AI-generated text is factually grounded in developer-provided source documents. It is not a RAG pipeline, web scraper, or agentic framework — it is a deterministic assert layer for document-intensive workflows.

---

## Commands

```bash
# Install for development (all optional extras)
pip install -e ".[dev,loaders,langchain]"

# Run fast suite (zero LLM calls — default for all dev work)
pytest -m "not llm and not loaders and not langchain" -x -q

# Run a single test
pytest tests/test_tier2.py::test_all_zero_scores_triggers_escalate_all_low_score -x -q

# Run real LLM integration tests (requires OPENAI_API_KEY or GOOGLE_API_KEY)
pytest -m llm --timeout=120 -q

# Run loaders tests (requires pip install -e ".[loaders]")
pytest -m loaders -q

# Run with coverage report
pytest -m "not llm and not loaders and not langchain" --cov=agentic_verifier --cov-report=term-missing
```

---

## Build Status (Session 5 — 2026-04-04)

**Phases 0–8 complete. Phase 9 is next. 52 tests passing.**

Last commit: `9b60e69` (post-review bug fixes)

| Phase | Content | Status |
|---|---|---|
| 0–8 | Scaffold, models, all tiers, chunker, classifier, builder | ✅ Done |
| 9 | `core/verifier.py` — `verify()`, `averify()`, `verify_batch()`, `verify_structured()` | Stub only — **START HERE** |
| 10 | `test_logging.py` — TDD #21/22/23 | Not started |
| 11 | `integrations/langchain.py` — `AgenticVerifierCallback` | Stub only |
| 12–13 | Integration tests + API boundary tests | Not started |
| 14 | Tier 1 calibration benchmark | Not started |
| 15 | README + PyPI publish | Not started |

To resume: read `plan/tasks.md` for task details, `plan/ORCHESTRATOR.md` for the execution strategy, `plan/engineering_design_update.md` §8 for the `verify()` pseudocode. Then say "continue from Phase 9".

---

## Execution Process Rules

These rules were established after a code review in session 5 found bugs caused by skipping them. **Do not shortcut these.**

### Strict Role Separation (ORCHESTRATOR.md §3)

Every implementation task requires **separate agent calls** for each role. Never combine into one:

```
1. Test Writer Agent   → writes RED test file, commits to main (isolation: none)
2. Coder Agent         → implements in worktree until GREEN (isolation: "worktree")
3. Code Reviewer Agent → reviews diff against spec (subagent_type: "Explore")
4. Fix Agent           → applies reviewer fixes if needed (isolation: "worktree")
5. Test Runner         → confirms GREEN after fixes
6. Git Commit Agent    → merges worktree to main
```

**Why:** Session 5 collapsed all 6 roles into 1 combined agent per module. The Code Reviewer was skipped entirely. This allowed 5 bugs to reach main undetected (wrong default values, incorrect routing logic, missing exception types, hardcoded field values).

### Code Reviewer Is Mandatory

After every Coder Agent, dispatch a Code Reviewer (`subagent_type: "Explore"`) with:

- The `git diff` of the branch
- The relevant spec section from `plan/engineering_design_update.md`
- The critical constraints checklist from CLAUDE.md

Do not proceed to the next task if the reviewer returns `approved: false`.

### Worktree Isolation for Coders

Every Coder Agent call must use `isolation: "worktree"`. The Test Writer commits to `main` first; the Coder's worktree branches from that commit automatically.

### Parallel Dispatch — Count Before Sending

Before sending a message with multiple parallel `Agent` calls, count the expected agents and verify all are present. Session 5 missed Worker F (Result Builder) in the first parallel wave, causing it to run solo later instead of in parallel.

### Context Injection — No Placeholder Left Behind

Every agent prompt must have all `[paste ...]` markers replaced with verbatim spec content before dispatching. An unresolved placeholder causes the agent to hallucinate silently (ORCHESTRATOR.md §9).

---

## Pipeline Architecture

The pipeline is a sequential 4-tier chain. `core/verifier.py` (stub only — Phase 9 implements it) is the orchestrator.

```
verify(claim, sources)
    │
    ├── Tier 0  core/classifier.py          parse_and_classify(claim)
    │           → list[ClassifiedAtom]      zero-cost rules, decimal-safe regex
    │
    ├── chunker loaders/chunker.py          chunk_sources(ctx)
    │           → list[Chunk]               sliding window if source > max_source_tokens
    │
    ├── Tier 1  tiers/tier1_authenticity.py check_fuzzy(evidence, chunks, threshold)
    │           gate only — raises or passes  rapidfuzz partial_token_set_ratio
    │           NEVER produces a terminal result
    │
    ├── Tier 2  tiers/tier2_semantic.py     route_claim(ctx, chunks)
    │           → Tier2Result               BM25Okapi — 3 routing branches (see below)
    │
    └── Tier 3  tiers/tier3_evaluation.py   evaluate(ctx, chunks)
                → Tier3ResponseModel        litellm.completion, 2-attempt retry
                → ResultBuilder             → VerificationResult (public output)
```

### Tier 2 Routing Branches

| Condition | Branch | Action |
|---|---|---|
| `score >= 0.85` | A — `SKIP_LLM_HIGH_CONFIDENCE` | Return VERIFIED, no LLM call |
| `0.01 < score < 0.85` | B — `ESCALATE_TO_LLM` | Send top-k chunks to Tier 3 |
| `score <= 0.01` and `raw_score >= 0` | C — `ESCALATE_ALL_LOW_SCORE` | Send all chunks (capped at `top_k * 3`, document order) |

The `raw_score >= 0` guard in Branch C is intentional: BM25Okapi returns negative scores on very small corpora (IDF artefact) — those fall through to Branch B instead.

### Key Mapping Rules (ResultBuilder)

- `Tier3ResponseModel.factual_consistency_score` is **0–100**; `VerificationResult.factual_consistency_score` is **0.0–1.0**. `ResultBuilder` divides by 100.
- `Neutral` entailment → **always** `"UNVERIFIABLE"`, `is_valid=False`. Never promoted regardless of coverage %.
- `sources_used` is filtered against `ctx.original_sources` — hallucinated `source_id`s are scrubbed.

---

## Module Ownership Rules

These prevent circular imports and are load-bearing:

| Rule | Detail |
|---|---|
| `Chunk` defined in `loaders/chunker.py` | NOT in `models/internal.py` — would create circular import |
| `models/internal.py` imports `Chunk` only under `TYPE_CHECKING` | Use string annotations at runtime |
| All tier modules import `Chunk` under `TYPE_CHECKING` | Same pattern throughout |

---

## Data Model Hierarchy

```
Public (models/result.py):       Source, AtomicClaimResult, VerificationResult
Internal (models/internal.py):   VerificationContext, SharedCostTracker, ClassifiedAtom,
                                  RoutingDecision, Tier2Result, ClaimInput
Tier 3 (models/tier3.py):        Tier3ResponseModel + sub-models (Pydantic v2 BaseModel)
Chunker (loaders/chunker.py):    Chunk  ← defined here, not in models/
```

`VerificationContext` is the per-call state bag passed through all tiers, constructed once per `verify()` call.

---

## Critical Implementation Details

**`SharedCostTracker`** — soft cap enforced *after* LLM call completes; triggering call is already billed. Cap check runs inside `threading.Lock`. Default `max_spend=float('inf')` — **do not change to a numeric default**.

**`_boundary_id`** — `secrets.token_hex(6)`, 12 hex chars, 48-bit entropy. Set at `VerificationContext` construction. `render_prompt` reads `ctx._boundary_id` — it never generates a new one.

**Tier 1 algorithm** — uses `rapidfuzz.fuzz.partial_token_set_ratio` (deliberate deviation from spec's `partial_ratio`). This is a known, preserved decision: `partial_ratio` scored too low on dropped-filler-word cases. Do not revert to `partial_ratio` without re-running the calibration benchmark (Phase 14).

**`parse_response`** — strips markdown fences before `model_validate_json`. Retry catches both `pydantic.ValidationError` and `ValueError` (latter covers `json.JSONDecodeError`).

**`auto_chunk=False`** — recommended for large-context models (Gemini 1.5 Pro, Claude 3.5+). BM25 can silently drop low-scoring chunks that contain negating context ("Lost Context Problem").

**Exception contract** — `verify()` / `averify()` are fail-loud (all exceptions propagate except `ParseError` → returned as `status="PARSE_ERROR"`). `verify_batch_async()` is fail-contained (all exceptions absorbed per item). This asymmetry is intentional — see `plan/engineering_design_update.md` §8.

---

## Known Open Issues (from session 5 code review)

These were identified but not yet fixed — address before shipping:

| # | File | Issue |
|---|---|---|
| 1 | `tiers/tier3_evaluation.py` | `evaluate_async()` has 0% test coverage — no async test exists |
| 2 | `tiers/tier2_semantic.py` | Branch C with vocabulary overlap but score ≤ 0.01 has no test |
| 3 | `tiers/tier1_authenticity.py` | `check_fuzzy()` with empty `chunks=[]` has no test |
| 4 | `models/builder.py` | `build_lexical_pass()` with empty `matched_chunks=[]` has no test |
| 5 | `core/classifier.py` | Line 36 (`return []` after empty split) has no test reaching it |

---

## Test Structure

```text
tests/
├── test_exceptions.py      # Phase 1
├── test_log.py             # Phase 1
├── test_models.py          # Phase 2 — TDD #15 (VerificationContext defaults)
├── test_classifier.py      # Phase 3 — decimal-safe split, inferential signals
├── test_chunker.py         # Phase 4 — char offsets, overlap guard, sliding window
├── test_helpers.py         # Phase 4 — @pytest.mark.loaders (skipped by default)
├── test_tier1.py           # Phase 5 — fuzzy match boundaries
├── test_tier2.py           # Phase 6 — BM25 routing branches, Branch C cap
├── test_evaluation.py      # Phase 7 — render_prompt, parse_response, retry loop
├── test_builder.py         # Phase 8 — score division, Neutral mapping, page_hint
└── integration/            # Phase 12+ — @pytest.mark.llm (requires real API keys)
```

Fast Suite runs in ~5 seconds. `litellm.completion` and `litellm.acompletion` are always mocked in Fast Suite tests via `pytest-mock`.

---

## Using a Local LLM

The library uses `litellm` which supports Ollama natively. No code changes needed:

```bash
ollama pull deepseek-r1:7b   # or deepseek-r1:14b, deepseek-coder-v2:16b
ollama serve                  # starts on http://localhost:11434
```

```python
result = verify(claim="...", sources=[...], model="ollama/deepseek-r1:7b", max_spend=0.0)
```

Local models don't support `response_format=Tier3ResponseModel`. `parse_response` handles this via its fence-stripping fallback. For Phase 12 integration tests, increase `--timeout` to 300s.
