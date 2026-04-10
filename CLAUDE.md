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

## Build Status (Session 10 — 2026-04-10)

**All phases 0–17 complete (code tasks). 118 fast tests passing. Real Suite 18/18 green against qwen3:30b. Compat Ollama baseline: 2/3 PASS (qwen3.5:9b pending). NIM baseline T-72 COMPLETE: 7/16 PASS cs01. Gemini pending.**

Last commit: `b497a34` (T-72: NIM compat baseline — 9 new models, 3 new adapters, 118 fast tests)

| Phase | Content | Status |
| --- | --- | --- |
| 0–15 | All phases | ✅ Done |
| Real Suite | `tests/integration/test_real_suite.py` — 18 fixtures | ✅ 18/18 PASS against `ollama/qwen3:30b` |
| Phase 16 | Multi-model compatibility — T-60 to T-66 | ✅ Done (2026-04-08) |
| Phase 17 | Multi-endpoint compat testing — T-67 to T-74 | ✅ Code done; ✅ T-71 Ollama 4/4×2 PASS; ✅ T-72 NIM 7/16 cs01 PASS; ⏳ Gemini pending |

### Phase 17 Summary (Session 10 — NIM RUNS COMPLETE)

#### T-72 NIM baseline (Session 10 — 2026-04-10)

Four fixes applied during NIM runs:

1. **`litellm.completion_cost()` raises for unknown NIM models** — `nvidia_nim/` models are absent from litellm's pricing registry. Fix: wrapped in `try/except` in both sync and async paths; cost = 0.0 for unknown models.
2. **`NIM_THINKING_ADAPTER`** — clean reasoning_content fallback for NIM thinking models (kimi-k2, gpt-oss, deepseek-v3.2) without Ollama-specific `extra_body.options` / `keep_alive`.
3. **`NEMOTRON_NIM_ADAPTER`** — Nemotron-Super hangs without `extra_body.chat_template_kwargs + reasoning_budget`. New adapter injects these.
4. **`JSON_OBJECT_ADAPTER`** — phi-4-mini only supports `json_object`, not `json_schema`. New adapter downgrades `response_format` type.
5. Fixed model ID typos: `deepseek-v3.2` (dot not underscore), `granite-3.3-8b-instruct` (dots).
6. Expanded NIM registry from 7 → 16 models (added deepseek-v3.2, nemotron-super, gemma4, kimi-k2, mistral-small-4, phi-4-mini, granite-3.3-8b, gpt-oss-120b, minimax-m2.5).

**NIM cs01 results:**

| Model | cs01 | Notes |
| --- | --- | --- |
| `llama-3.3-70b-instruct` | ✅ 4/4 full suite | |
| `deepseek-v3.2` | ✅ | ~47s |
| `gemma-4-31b-it` | ✅ | ~10s |
| `mistral-small-4-119b` | ✅ | ~5s |
| `gpt-oss-120b` | ✅ | ~5s |
| `minimax-m2.5` | ✅ | ~10s |
| `kimi-k2-thinking` | ✅ | ~138s (thinking model) |
| `nemotron-super-120b` | ⏳ server timeout | Code fix ready; awaiting server availability |
| `phi-4-mini-instruct` | ⏳ server timeout | Code fix ready; awaiting server availability |
| `granite-3.3-8b-instruct` | ⏳ server timeout | Code fix ready; awaiting server availability |
| `deepseek-r1` | ❌ 410 Gone | EOL 2026-01-26 |
| `nemotron-70b` | ❌ 404 | No account access |
| `qwen25-72b` | ❌ 404 | Unavailable |
| `mixtral-8x22b` | ❌ timeout | Server unresponsive |
| `nemotron-340b` | ❌ UnsupportedParams | No `response_format` support |
| `mistral-small-24b` | ❌ 400 | `json_schema` not supported |

Fast suite: **118 tests** passing.

#### Session 9 adapter fixes (OLLAMA_ADAPTER `build_kwargs`)

Three bugs found during Ollama runs:

1. **litellm routes `ollama/` → `/api/generate`** instead of `/api/chat` — remapped to `ollama_chat/`.
2. **Thinking-capable models exhaust 4K default context** — `_ollama_build_kwargs` sets `num_ctx=16384`.
3. **`keep_alive=300`** added to hold model in VRAM during sequential test calls.

Run compat suite (keys auto-determine which models execute):

```bash
pytest -m compat -v --timeout=300 -p no:cov
```

Run full real suite against a NIM model:

```bash
LLM_TEST_MODEL=nvidia_nim/meta/llama-3.3-70b-instruct \
  pytest tests/integration/test_real_suite.py -m llm -v --timeout=300 -p no:cov
```

### Phase 16 Summary (Session 7 — COMPLETE)

Phase 16 delivered 7 tasks (T-60 through T-66) adding formal multi-model compatibility:

- **T-60** — `tests/test_adapters.py`: 17 RED tests for adapter registry (routing + post_process)
- **T-61** — `adapters/registry.py`: `ModelAdapter` dataclass, `get_adapter()` prefix lookup, 5 concrete adapters (DEFAULT, OLLAMA, OPENAI_REASONING, ANTHROPIC, GOOGLE)
- **T-62** — `tier3_evaluation.py`: wired registry in; removed `_extra_kwargs()` + `think=False`; `parse_response(response, model)` now routes via adapter
- **T-63** — Fast suite gap tests: think-tag stripping, `choices=[]` → `ParseError`, `auto_chunk=False`, async cost cap; 3 existing test reliability fixes
- **T-64** — Integration contract corrections: T-36/T-43 relaxed to soft `VERIFIED or UNVERIFIABLE`; T-39 softened to `!= "VERIFIED"`; T-49 `sources_used` restored as soft assertion
- **T-65** — `conftest.py` `loader_fixtures`: generates `sample.pdf` + `sample.docx` on-demand via fpdf2/python-docx
- **T-66** — Exponential backoff wrappers (`_acompletion_with_backoff`, `_completion_with_backoff`) for transient LiteLLM errors (1s base, 2× multiplier, 30s max, 3 attempts)

Run full integration suite:

```bash
pytest tests/integration/ -m llm -v --timeout=300
```

---

## Execution Process Rules

These rules were established after a code review in session 5 found bugs caused by skipping them. **Do not shortcut these.**

### Strict Role Separation (ORCHESTRATOR.md §3)

Every implementation task requires **separate agent calls** for each role. Never combine into one:

```text
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

```text
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
                → adapters/registry.py      get_adapter(model) → pre_call_kwargs + post_process
                → Tier3ResponseModel        litellm.completion, 2-attempt retry
                → ResultBuilder             → VerificationResult (public output)
```

### Tier 2 Routing Branches

| Condition | Branch | Action |
| --- | --- | --- |
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
| --- | --- |
| `Chunk` defined in `loaders/chunker.py` | NOT in `models/internal.py` — would create circular import |
| `models/internal.py` imports `Chunk` only under `TYPE_CHECKING` | Use string annotations at runtime |
| All tier modules import `Chunk` under `TYPE_CHECKING` | Same pattern throughout |

---

## Data Model Hierarchy

```text
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

**`parse_response(response, model)`** — routes via `get_adapter(model).post_process()`. OLLAMA_ADAPTER strips `<think>` tags (rfind-based; regex fallback), falls back to `reasoning_content` if content is empty. DEFAULT adapter strips markdown fences. Retry catches `pydantic.ValidationError`, `ValueError`, and `IndexError` (the last covers `choices=[]` edge case).

**`auto_chunk=False`** — recommended for large-context models (Gemini 1.5 Pro, Claude 3.5+). BM25 can silently drop low-scoring chunks that contain negating context ("Lost Context Problem").

**Exception contract** — `verify()` / `averify()` are fail-loud (all exceptions propagate except `ParseError` → returned as `status="PARSE_ERROR"`). `verify_batch_async()` is fail-contained (all exceptions absorbed per item). This asymmetry is intentional — see `plan/engineering_design_update.md` §8.

---

## Known Open Issues

All Phase 16 tasks are complete. No open issues. Remaining deferred items:

- **T-59** — PyPI publish: deferred, requires package registry credentials.
- **Real Suite re-run on alternative models** — contracts now use hybrid hard/soft structure (T-64) but have only been validated against `ollama/qwen3:30b`. A run against GPT-4o or Claude 3.5 Sonnet is recommended before publishing.

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
├── test_adapters.py        # Phase 16 (T-60) — adapter prefix routing, post_process, pre_call_kwargs
├── fixtures/               # Phase 16 (T-65) — sample.pdf, sample.docx (generated, gitignored)
│   └── scaffold_fixtures.py  # Run once to generate fixture files
└── integration/            # Phase 12+ — @pytest.mark.llm (requires real API keys)
```

Fast Suite runs in ~5 seconds. `litellm.completion` and `litellm.acompletion` are always mocked in Fast Suite tests via `pytest-mock`.

---

## Using a Local LLM

The library uses `litellm` which supports Ollama natively. No code changes needed:

```bash
ollama pull qwen3:14b    # ~9GB Q4_K_M, fits in 15GB RAM — recommended for integration tests
ollama serve             # starts on http://localhost:11434
```

```python
result = verify(claim="...", sources=[...], model="ollama/qwen3:14b", max_spend=0.0)
```

**Ollama thinking mode (Phase 16 complete)** — Thinking-capable Ollama models (qwen3, DeepSeek-R1, Gemma 4, Kimi K2, LFM2.5, GPT-OSS, etc.) emit `<think>...</think>` tags or drop `content` when thinking is active. `OLLAMA_ADAPTER` in `adapters/registry.py` handles this automatically: strips `<think>` tags (rfind-based, regex fallback), falls back to `reasoning_content` if content is empty. Models reason freely — `think=False` was removed in T-62.

Ollama supports `response_format` (structured output via JSON schema grammar). `parse_response()` uses the fence-stripping + `<think>` tag stripping fallback for any remaining edge cases.

For the Real Suite integration tests, use `--timeout=300` (qwen3:30b takes 30–90s per call).
