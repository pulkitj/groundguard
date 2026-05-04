# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Project Is

**groundguard** is a Python middleware library (MIT, LLM-agnostic) that verifies AI-generated text is factually grounded in developer-provided source documents. It is not a RAG pipeline, web scraper, or agentic framework — it is a deterministic assert layer for document-intensive workflows.

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

### Code Reviewer Prompt — Required Elements (session 6 correction)

The reviewer prompt **must** include all four of these or the review is invalid:

1. **Verbatim `git diff`** — run `git diff HEAD~1` (or `git diff main...<branch>`) and paste the full output inline. Do not ask the reviewer to read files instead.
2. **Verbatim spec section** — paste the exact section from `plan/engineering_design_v4.md` for the module. Do not paraphrase or summarise it.
3. **Role 4 output format exactly** — the reviewer must return:

   ```json
   {
     "approved": true,
     "issues": [
       {"severity": "blocking"|"advisory", "file": "...", "line_hint": "...", "description": "...", "fix": "..."}
     ]
   }
   ```

   Do not use alternate schemas (`findings`, `constraint_checks`, etc.) — the `severity` field is load-bearing: `"blocking"` triggers a Fix Agent, `"advisory"` does not.
4. **No test execution** — the reviewer is read-only (`Explore`). Do not ask it to run `pytest`. That is the Test Runner's job (Role 5).

**Why:** In Phase 21 the reviewer prompt omitted the git diff, paraphrased the spec instead of pasting it verbatim, used a non-standard output schema, and asked the reviewer to run tests. The review still passed because the implementation happened to be correct — but the process was non-compliant and would have failed to catch spec deviations reliably.

### Worktree Isolation for Coders

Every Coder Agent call must use `isolation: "worktree"`. The Test Writer commits to `main` first; the Coder's worktree branches from that commit automatically.

### Parallel Dispatch — Count Before Sending

Before sending a message with multiple parallel `Agent` calls, count the expected agents and verify all are present. Session 5 missed Worker F (Result Builder) in the first parallel wave, causing it to run solo later instead of in parallel.

### Context Injection — No Placeholder Left Behind

Every agent prompt must have all `[paste ...]` markers replaced with verbatim spec content before dispatching. An unresolved placeholder causes the agent to hallucinate silently (ORCHESTRATOR.md §9).

---

## Token Efficiency Protocols

### Before reading any file

Check `CODEBASE_ANALYSIS.md` (same directory as this file) first.
It contains the full dependency graph, module contracts, implementation
status, and red flags. Read it instead of source files to orient yourself.
Only read a source file when you are about to act on it.

### Native tool rules

- **Read tool:** State which file and which section before reading.
  Use line ranges when the section is known. If already read this
  session, use what is in context — do not re-read.
- **Grep tool:** Use `output_mode: "files_with_matches"` first.
  Only switch to content mode when you need the actual lines.
- **Agent tool:** Only spawn agents when: single named role, all
  context injected inline, mutually exclusive file scope.

### Bash command rules

Always use the quiet flags below. Never use the verbose form.

```
pytest (fast suite)  →  pytest -m "not llm and not loaders and not langchain" -x -q --tb=short --no-header
pytest (single test) →  pytest <test_path> -x -q --tb=short
pytest (compat)      →  pytest -m compat -v --timeout=300 -p no:cov
pytest (llm)         →  pytest -m llm --timeout=120 -q
pip install          →  pip install -q -e ".[dev,loaders,langchain]"
git log              →  git log --oneline -10
git diff             →  git diff --stat
git status           →  git status -s
grep                 →  grep -rn "term" . | head -20
find                 →  find . -name "*.py" | grep -v __pycache__ | head -20
```

### Document updates after a discussion

Do NOT immediately read all plan documents.

1. Extract decisions from the conversation (they are already in context)
2. Run `grep -n "^#" <filename>` to get section headings cheaply
3. Read only the specific sections that need updating
4. Edit targeted sections — never rewrite whole documents

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
pytest -m "not llm and not loaders and not langchain" --cov=groundguard --cov-report=term-missing
```

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

## Test Structure

```text
tests/
├── test_exceptions.py      # Phase 1
├── test_log.py             # Phase 1
├── test_models.py          # Phase 2 — TDD #15 (VerificationContext defaults) + Phase 21 (profile fields)
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
