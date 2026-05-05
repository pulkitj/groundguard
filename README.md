# groundguard

**Your LLM will hallucinate. groundguard catches it before it ships.**

A drop-in verification layer that checks whether AI-generated text is actually supported by your source documents — before it reaches users.

```
pip install groundguard
```

---

## The problem

RAG pipelines, document Q&A, and agentic workflows all share one failure mode: the model confidently states something that isn't in the source. You won't know until a lawyer, a clinician, or a customer finds it.

groundguard sits between your pipeline output and your application. It reads your source documents, checks the claim, and returns a structured verdict — so you can gate on it, log it, or retry.

---

## Quickstart

```python
from groundguard import verify
from groundguard.models.result import Source

sources = [
    Source(content="Q3 revenue was $5 million.", source_id="report.pdf"),
]

result = verify(
    claim="Revenue was $5M in Q3.",
    sources=sources,
    model="gpt-4o-mini",
)

assert result.is_valid          # True
print(result.status)            # "VERIFIED"
print(result.factual_consistency_score)  # 0.0–1.0
```

That's it. No pipeline changes, no new infrastructure. Pass in your source documents and the claim to check — groundguard does the rest.

---

## What it returns

```python
@dataclass
class AtomicClaimResult:
    claim: str
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PARSE_ERROR"]
    factual_consistency_score: float   # 0.0–1.0
    verification_method: str           # "tier2_lexical" | "tier3_llm"
    citation: Citation | None          # excerpt + char offsets; always set on VERIFIED
    is_valid: bool

@dataclass
class VerificationResult:
    is_valid: bool
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE",
                    "PARSE_ERROR", "SKIPPED_DUE_TO_COST", "ERROR"]
    atomic_claims: list[AtomicClaimResult]
    factual_consistency_score: float
    sources_used: list[str]            # source_ids that supported the verdict
    rationale: str
    offending_claim: str | None        # first CONTRADICTED sentence
    total_cost_usd: float
```

Every `VERIFIED` result has a non-null `citation` with the exact excerpt and character offsets. `CONTRADICTED` results include the offending sentence. `UNVERIFIABLE` means the source documents don't have enough information to decide — not that the claim is false.

---

## Core API

### Single claim

```python
from groundguard import verify, averify

# Synchronous
result = verify(claim="...", sources=[...], model="gpt-4o-mini")

# Async — BM25 dispatched to executor, non-blocking
result = await averify(claim="...", sources=[...], model="gpt-4o-mini")
```

### Full answer grounding (`verify_answer`)

Check whether a complete LLM response is supported by the retrieved documents:

```python
from groundguard import verify_answer, averify_answer

result = verify_answer(
    output="The acquisition closed in Q2 and the FTC approved it unconditionally.",
    sources=retrieved_docs,
    model="gpt-4o-mini",
)
# result.is_grounded, result.score, result.status
```

### Multi-claim analysis (`verify_analysis`)

Extract claims from a longer text and verify each one independently:

```python
from groundguard import verify_analysis

result = verify_analysis(
    analysis="Revenue grew 30% YoY. Headcount declined by 12%.",
    sources=company_filings,
    model="gpt-4o-mini",
)
# supported / (supported + contradicted) — UNVERIFIABLE excluded from denominator
```

### Legal clause verification (`verify_clause`)

Decompose a contract clause into obligations, then verify each term against a source:

```python
from groundguard import verify_clause
from groundguard.loaders.legal import TermRegistry

registry = TermRegistry()
registry.define("Effective Date", "January 1, 2025")

result = verify_clause(
    clause="Payment is due within 30 days of the Effective Date.",
    sources=contract_sources,
    term_registry=registry,
)
```

Uses `STRICT_PROFILE` by default (faithfulness threshold 0.97, majority vote on, full audit trail).

### Batch verification

```python
from groundguard import averify_batch
from groundguard.models.internal import ClaimInput

inputs = [
    ClaimInput(claim="Claim A", sources=[...]),
    ClaimInput(claim="Claim B", sources=[...], model="claude-3-haiku-20240307"),
]

results = averify_batch(inputs=inputs, model="gpt-4o-mini", max_concurrency=5, max_spend=2.0)
```

Fail-contained: individual failures do not abort the batch. Items that hit the spend cap return `status="SKIPPED_DUE_TO_COST"`.

---

## Verification profiles

Tune strictness to your domain with a single parameter:

```python
from groundguard import verify_answer, STRICT_PROFILE, GENERAL_PROFILE, RESEARCH_PROFILE

# Legal / compliance / clinical — 0.97 threshold, majority vote, full audit trail
result = verify_answer(output, sources, profile=STRICT_PROFILE, model="gpt-4o-mini")

# General RAG / product — 0.80 threshold, single call, no audit
result = verify_answer(output, sources, profile=GENERAL_PROFILE, model="gpt-4o-mini")

# Research / exploratory — permissive threshold, wider BM25 retrieval
result = verify_answer(output, sources, profile=RESEARCH_PROFILE, model="gpt-4o-mini")
```

| Profile | Threshold | Majority vote | Audit trail | Top-k chunks |
| --- | --- | --- | --- | --- |
| `STRICT_PROFILE` | 0.97 | Yes (3 calls) | Full | 6 |
| `GENERAL_PROFILE` | 0.80 | No | Off | 3 |
| `RESEARCH_PROFILE` | 0.70 | No | Off | 4 |

Custom profiles are a frozen dataclass — create your own in one line.

---

## Production patterns

### Assert and raise

```python
from groundguard import assert_faithful, assert_grounded, GroundingError

# Raises GroundingError if output is not grounded
assert_faithful(llm_output, sources, model="gpt-4o-mini")

# Raises GroundingError if analysis contains unsupported claims
assert_grounded(analysis_text, sources, model="gpt-4o-mini")
```

### Retry until grounded

```python
from groundguard import verify_or_retry

verified_output = verify_or_retry(
    generator=lambda: chain.invoke({"query": "What was Q3 revenue?"}),
    sources=retrieved_docs,
    max_retries=3,
    model="gpt-4o-mini",
)
```

### Spend cap

```python
result = verify(
    claim="...",
    sources=[...],
    model="gpt-4o-mini",
    max_spend=0.50,   # soft USD cap — triggering call completes, then raises
)
```

---

## How it works

Every `verify()` call runs a 4-tier pipeline with no extra latency when the answer is obvious:

| Tier | What it does | Cost |
| --- | --- | --- |
| **0 — Classifier** | Splits claim into atomic sentences | Zero |
| **1 — Fuzzy gate** | Confirms agent-cited evidence appears in source | Zero |
| **2 — BM25 router** | Lexical similarity check — skips LLM if score ≥ 0.85 | Zero |
| **3 — LLM evaluator** | Sends top-k chunks + claim; Pydantic-validated verdict | LLM call |

Near-verbatim quotes never hit the LLM. Paraphrases and inferred claims do. The router decides automatically.

### Large-context models

```python
result = verify(
    claim="...",
    sources=[Source(content=long_document, source_id="contract.pdf")],
    model="gemini-2.0-flash",
    auto_chunk=False,   # pass full source — no BM25 chunking
)
```

Use `auto_chunk=False` when your model can fit the full document. By default, BM25 retrieves isolated chunks by keyword relevance — a negating clause in a low-scoring chunk (e.g., *"that acquisition was subsequently blocked by the FTC"*) may not reach Tier 3. `auto_chunk=False` eliminates this risk for large-context models.

---

## Local LLMs (no cloud required)

```bash
ollama pull qwen3:14b   # ~9GB, fits in 15GB RAM
ollama serve
```

```python
result = verify(
    claim="...",
    sources=[...],
    model="ollama/qwen3:14b",
    max_spend=0.0,   # local — free
)
```

Works out of the box — litellm handles the Ollama transport. Thinking-mode models (qwen3, DeepSeek-R1, etc.) are fully supported; `<think>` tags are stripped automatically.

---

## Source accumulation

When multiple agents or retrieval steps contribute sources, deduplicate and track provenance before verifying:

```python
from groundguard import verify_analysis
from groundguard.loaders.accumulator import SourceAccumulator

acc = SourceAccumulator()
acc.add(db_source, provenance="database_lookup", agent_id="agent_1")
acc.add(llm_source, provenance="llm_generated", is_llm_derived=True, agent_id="agent_2")

result = verify_analysis(agent_output, sources=acc.sources(), model="gpt-4o-mini")
```

`acc.sources()` returns a deduplicated `list[Source]` with provenance metadata. Inspect or filter before passing to any verification function.

---

## Cost estimation

Estimate cost before running, not after:

```python
from groundguard import estimate_verify_faithfulness_cost, estimate_verify_analysis_cost, CostEstimate

estimate: CostEstimate = estimate_verify_faithfulness_cost(
    output="The report shows revenue of $5M for Q3.",
    sources=retrieved_docs,
    model="gpt-4o-mini",
    return_breakdown=True,
)

print(f"Estimated: ${estimate.total_cost_usd:.4f}")
```

---

## Use by domain

**Legal / compliance** — verify contract clause obligations against source agreements using `verify_clause` + `TermRegistry`. Pinned term definitions are injected as sources automatically.

**Financial** — `verify_analysis` extracts and checks every numerical claim in a generated report. Tier 2.5 catches numeric conflicts (unit mismatches, magnitude errors, off-by-one) before the LLM call.

**Clinical / medical** — use `STRICT_PROFILE` for 0.97 faithfulness threshold and majority vote. Every sentence result gets a `VerificationAuditRecord` with the raw LLM output, confidence, and grounding source ID.

**RAG pipelines** — `assert_faithful` as a post-retrieval gate: if the generated answer isn't supported by the retrieved chunks, raise before the response is returned to the user.

---

## What groundguard is not

- A RAG pipeline or retrieval system (you provide the source documents)
- A web scraper or URL fetcher (sources must be developer-provided text)
- A general-purpose question-answering system
- A model fine-tuning or evaluation framework

---

## Optional extras

```bash
pip install groundguard[loaders]    # PDF and DOCX ingestion
pip install groundguard[langchain]  # LangChain callback integration
pip install groundguard[dev]        # pytest, coverage, asyncio test utilities
```

### Ingest PDFs and DOCX

```python
from groundguard.loaders.helpers import pdf_to_text, docx_to_text

text = pdf_to_text("contract.pdf")
sources = [Source(content=text, source_id="contract.pdf")]
result = verify_clause(clause="...", sources=sources)
```

### LangChain integration

```python
from groundguard.integrations.langchain import AgenticVerifierCallback

callback = AgenticVerifierCallback(model="gpt-4o-mini")
chain.invoke({"query": "What was Q3 revenue?"}, callbacks=[callback])
# Raises GroundingError if the chain output cannot be verified against source_documents
```

---

## License

MIT
