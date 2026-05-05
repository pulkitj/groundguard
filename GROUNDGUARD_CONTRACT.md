# GROUNDGUARD_CONTRACT.md

This document defines the execution contract for the `groundguard` library. It describes what the library guarantees unconditionally, what callers can configure, how invariants hold under composition, and what is explicitly out of scope.

---

## 1. Guarantees

These invariants always hold. If any of them would be violated, an exception is raised instead of returning a result.

**Citation on VERIFIED results**
Every `AtomicClaimResult` with `status="VERIFIED"` has a non-null `citation`. This is enforced by `_assert_citation_invariant` in `ResultBuilder` — a `VERIFIED` result with a null citation raises `InvariantError` before it can be returned.

**Non-negative cost**
`cost_usd` is never negative on any result. The `max_spend` cap is a *soft* cap: the triggering LLM call is allowed to complete and is billed before the cap fires. Subsequent calls in the same context are blocked.

**Per-call boundary ID**
Every `verify()` and `averify()` call generates a unique `boundary_id` of exactly 12 hex characters (48-bit entropy via `secrets.token_hex(6)`). The boundary ID is embedded in the prompt to prevent prompt-injection attacks that splice content across the boundary. It is never reused across calls.

**Majority vote call count**
When majority vote is active (triggered by a profile with `majority_vote=True` and a result score below `majority_vote_confidence_threshold`), exactly 3 LLM calls are made — never 1 or 2. The vote cannot complete with fewer.

**Tie-break conservatism**
A 1-1-1 split across 3 majority vote calls always yields `is_grounded=False` and `status="NOT_GROUNDED"`. Ties are never silently promoted to a positive verdict.

---

## 2. Configurables

These parameters are caller-controlled. They change behaviour but do not affect the invariants above.

**`profile`** — a `VerificationProfile` dataclass (preset: `GENERAL_PROFILE`, `STRICT_PROFILE`, or `RESEARCH_PROFILE`). Sets defaults for `faithfulness_threshold`, `tier2_lexical_threshold`, `bm25_top_k`, `majority_vote`, and `audit`. Explicit per-call parameters always override profile defaults.

**`faithfulness_threshold`** — float, 0.0–1.0. Minimum score for a result to be considered grounded. Explicit value beats `majority_vote_on_borderline` which beats the profile default (precedence enforced in `VerificationContext.__post_init__`).

**`max_spend`** — soft USD cap. Default `float('inf')` (no cap). The triggering call is billed; subsequent calls in the batch or context are blocked with `status="SKIPPED_DUE_TO_COST"`.

**`model`** — any litellm model string (e.g. `"gpt-4o-mini"`, `"ollama/qwen3:14b"`, `"gemini/gemini-2.0-flash"`). Passed through to litellm without transformation.

**`api_base`** — passed to litellm for custom endpoints (e.g. local Ollama, Azure deployments).

**`auto_chunk`** — default `True`. When `True`, long sources are split by a sliding-window chunker and BM25 retrieves the top-k chunks for Tier 3. When `False`, full source content is forwarded to Tier 3 without chunking — recommended for large-context models to avoid the Lost Context Problem (negating clauses in low-scoring chunks not reaching the LLM).

---

## 3. Invariants Under Composition

These invariants hold when combining multiple groundguard APIs in a single workflow.

**`SourceAccumulator` is opt-in**
`SourceAccumulator` is not wired into the pipeline. No verification function accepts a `SourceAccumulator` directly — callers always call `.sources()` explicitly and pass the resulting `list[Source]`:

```python
acc = SourceAccumulator()
acc.add(db_source, provenance="database_lookup", agent_id="agent_1")
acc.add(llm_source, provenance="llm_generated", is_llm_derived=True, agent_id="agent_2")
result = verify_analysis(agent_output, sources=acc.sources(), model="gpt-4o-mini")
```

This keeps the public API stable and lets callers inspect, filter, or log the source list before verification runs.

**`verify_clause` + `TermRegistry`**
When a `TermRegistry` is provided to `verify_clause`, term definitions are injected as pinned `Source` objects into the sources list *before* `verify()` is called. Term resolution runs before Tier 0, so pinned definitions affect the atom count and can alter routing decisions.

**`averify_batch` failure isolation**
`averify_batch` is fail-contained per item. An exception in one item (including `VerificationCostExceededError`) does not abort the batch. Items that hit the spend cap return `status="SKIPPED_DUE_TO_COST"`. All other per-item exceptions return `status="ERROR"`. The shared `SharedCostTracker` applies across the entire batch.

**Parameter precedence**
`explicit call params > ctx.majority_vote_on_borderline > profile defaults`. This precedence is enforced in `VerificationContext.__post_init__` and cannot be overridden by downstream tiers.

**`verify()` / `averify()` vs. `averify_batch`**
`verify()` and `averify()` are fail-loud: all exceptions propagate to the caller except `ParseError`, which is returned as `status="PARSE_ERROR"`. `averify_batch` is fail-contained. This asymmetry is intentional — single-call users get explicit error signals; batch callers get partial results without a full abort.

---

## 4. Undefined Behaviour

The following scenarios are not guaranteed to work correctly and may change across versions without a deprecation notice.

**Models that ignore `response_format`**
Groundguard requests structured JSON output via `response_format`. If a model ignores the schema (returning free text instead), the retry loop attempts to parse the output using fence-stripping and Pydantic validation. This is a best-effort fallback — it is not guaranteed to succeed on all model/prompt combinations.

**Tier 2.5 on non-English text**
The numerical conflict detector (`tier25_preprocessing`) uses regex patterns calibrated on English-language text (digits, `%`, `$`, metric suffixes `M/B/K`). Behaviour on non-English numerals or currency formats is undefined.

**BM25 on single-sentence sources**
BM25Okapi can return negative scores on very small corpora due to IDF artefacts. On single-sentence sources, routing may fall through to Branch B (`ESCALATE_TO_LLM`) even when Branch C (`ESCALATE_ALL_LOW_SCORE`) would be more appropriate. Use `auto_chunk=False` on very short sources to avoid this edge case.
