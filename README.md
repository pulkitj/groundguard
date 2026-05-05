# groundguard

[![PyPI](https://img.shields.io/pypi/v/groundguard.svg)](https://pypi.org/project/groundguard)
[![Python](https://img.shields.io/pypi/pyversions/groundguard.svg)](https://pypi.org/project/groundguard)
[![CI](https://github.com/pulkitj/groundguard/actions/workflows/ci.yml/badge.svg)](https://github.com/pulkitj/groundguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLMs got significantly better at not hallucinating. They didn't get better at knowing when they do.**

groundguard is a verification layer that checks whether AI-generated output is actually grounded in your source documents — before it reaches users.

```bash
pip install groundguard
```

---

## The problem is structural, not a bug to be patched

You're not asking your model to answer from memory. You're building a pipeline:

```text
User query
  → Retrieve relevant documents from your knowledge base
  → (Optional) Summarize, transform, or route across agents
  → Generate the final answer
  → User
```

By the time the generator runs, it has never seen the original documents. It works from whatever was passed to it — a summary, a transformed excerpt, a previous agent's output. If that intermediate step dropped a qualifier, rounded a number, or lost a clause, the generator has no way to detect it. It produces a confident, well-formed answer regardless.

**Prompting harder doesn't fix this.** The generator cannot compare its output to your original sources — it doesn't have them. Verification has to happen independently, after generation, with the original documents in scope.

That's what groundguard does.

---

## Quickstart

```python
from groundguard import verify_answer
from groundguard.models.result import Source

# Original documents from your retrieval step
sources = [
    Source(content="Q3 revenue was $4.8 million, down 3% YoY.", source_id="q3_report.pdf"),
]

# Generated answer you want to verify
result = verify_answer(
    output="Q3 revenue came in at $5 million, roughly flat year-over-year.",
    sources=sources,
    model="gpt-4o-mini",
)

print(result.is_grounded)    # False
print(result.status)         # "NOT_GROUNDED"
# $5M vs $4.8M — caught before it reaches the user
```

No pipeline changes. No new infrastructure. Pass your source documents and the generated output — groundguard returns a structured verdict.

---

## Where it fits

groundguard is a backward pass. RAG is a forward pass.

```text
RAG:           sources → generate
groundguard:   output → verify against sources
```

They solve different halves of the same problem. RAG helps your model generate correctly. groundguard proves it did. In a typical pipeline, groundguard belongs here:

```text
Retrieval  →  [Optional: Summarize / Transform / Route]  →  Generation  →  groundguard  →  User
     ↑                                                                           ↓
 original sources  ←——————————————————————————————— verify against these
```

groundguard is the only component in your stack that sees both the original source documents and the final generated output simultaneously.

---

## Isn't this just an LLM checking an LLM?

It's a fair objection. Here's the honest answer.

**Most verifications never reach an LLM.** The BM25 routing tier resolves roughly 60–70% of claims lexically. Direct quotes, numbers, dates, and close paraphrases are verified without a single token of LLM cost. The LLM runs only when lexical similarity is ambiguous.

**The verifying LLM is constrained to your documents.** It isn't asked "is this true?" — it's asked "is this supported by these specific paragraphs?" It cannot draw on training knowledge to fill gaps. Either grounding exists in your sources or the result is `UNVERIFIABLE`. That's a fundamentally more constrained and reliable task than open-world fact-checking.

**The generator and verifier share no context.** Different prompt, different role, optionally a different model entirely. The generator is optimized for fluency and helpfulness. The verifier is optimized for strict factual grounding. This is separation of concerns — the same reason you have code review even when both engineers are capable.

**Ties are conservative.** When majority vote runs and the score splits 1-1-1, the verdict is `NOT_GROUNDED`. Uncertainty never silently resolves to a positive result.

---

## API

### Verify a complete answer

Check whether an LLM response is supported by your retrieved documents:

```python
from groundguard import verify, averify, verify_answer, averify_answer

# Synchronous
result = verify_answer(
    output="The acquisition closed in Q2 and the FTC approved it unconditionally.",
    sources=retrieved_docs,
    model="gpt-4o-mini",
)

# Async
result = await averify_answer(output=..., sources=..., model="gpt-4o-mini")
```

### Verify a multi-claim analysis

Extract every factual claim from longer text and verify each one independently:

```python
from groundguard import verify_analysis

result = verify_analysis(
    analysis="Revenue grew 30% YoY. Headcount declined by 12%. The Singapore office was closed.",
    sources=company_filings,
    model="gpt-4o-mini",
)
# result.grounded_units, result.total_units
# result.unit_results — per-claim breakdown with citation offsets
```

### Verify a legal clause

Decompose a contract clause into obligations and verify each term against source agreements:

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

Uses `STRICT_PROFILE` by default (0.97 threshold, majority vote, full audit trail per atom).

### Batch verification

```python
from groundguard import averify_batch
from groundguard.models.internal import ClaimInput

inputs = [
    ClaimInput(claim="Claim A", sources=[...]),
    ClaimInput(claim="Claim B", sources=[...], model="claude-3-haiku-20240307"),
]

results = averify_batch(inputs=inputs, model="gpt-4o-mini", max_concurrency=5, max_spend=2.00)
```

Fail-contained: one item failing does not abort the batch. Items that exceed the spend cap return `status="SKIPPED_DUE_TO_COST"`.

---

## What it returns

```python
@dataclass
class AtomicClaimResult:
    claim: str
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PARSE_ERROR"]
    factual_consistency_score: float    # 0.0–1.0
    verification_method: str            # "tier2_lexical" | "tier25_numerical" | "tier3_llm"
    citation: Citation | None           # excerpt + char offsets; always non-null on VERIFIED
    is_valid: bool
```

Every `VERIFIED` result carries a `citation` with the exact excerpt and character offsets — a pointer, not just a score. `CONTRADICTED` results include the offending sentence. `UNVERIFIABLE` means your source documents don't contain enough information to decide — not that the claim is false.

---

## Verification profiles

```python
from groundguard import verify_answer, STRICT_PROFILE, GENERAL_PROFILE, RESEARCH_PROFILE

result = verify_answer(output, sources, profile=STRICT_PROFILE, model="gpt-4o-mini")
```

| Profile | Threshold | Majority vote | Audit trail | BM25 top-k |
| --- | --- | --- | --- | --- |
| `STRICT_PROFILE` | 0.97 | Yes (3 calls) | Full | 6 |
| `GENERAL_PROFILE` | 0.80 | No | Off | 3 |
| `RESEARCH_PROFILE` | 0.70 | No | Off | 4 |

Custom profiles are a frozen dataclass:

```python
from groundguard.profiles import VerificationProfile

MY_PROFILE = VerificationProfile(
    name="clinical",
    faithfulness_threshold=0.95,
    tier2_lexical_threshold=1.5,
    bm25_top_k=5,
    majority_vote=True,
    audit=True,
)
```

---

## Production patterns

### Gate before responding

```python
from groundguard import assert_faithful, GroundingError

try:
    assert_faithful(llm_output, sources, model="gpt-4o-mini")
    return llm_output           # verified — safe to send
except GroundingError as e:
    log.warning("ungrounded output blocked", extra={"reason": str(e)})
    return fallback_response
```

### Retry until grounded

```python
from groundguard import verify_or_retry

verified_output = verify_or_retry(
    generator=lambda: chain.invoke({"query": user_query}),
    sources=retrieved_docs,
    max_retries=3,
    model="gpt-4o-mini",
)
```

### Spend cap

```python
result = verify(claim="...", sources=[...], model="gpt-4o-mini", max_spend=0.50)
# Soft cap: the triggering call completes and is billed; subsequent calls are blocked.
```

### Estimate cost before running

```python
from groundguard import estimate_verify_faithfulness_cost

estimate = estimate_verify_faithfulness_cost(
    output=llm_output,
    sources=retrieved_docs,
    model="gpt-4o-mini",
)
print(f"Estimated: ${estimate.total_cost_usd:.4f}")
```

---

## How it works

Every `verify()` call runs a sequential pipeline. Cheaper gates run first — the LLM is only called when earlier tiers can't resolve the claim:

| Tier | What runs | Resolves when |
| --- | --- | --- |
| **0 — Classifier** | Splits claim into atomic sentences | Always |
| **1 — Fuzzy gate** | Checks cited evidence appears verbatim in source | Direct quotes, near-verbatim excerpts |
| **2 — BM25 router** | Lexical similarity; skips LLM if score ≥ 0.85 | Clear matches and clear misses |
| **2.5 — Numerical detector** | Catches unit/magnitude conflicts before LLM | Numbers, percentages, currency figures |
| **3 — LLM evaluator** | Constrained to top-k source chunks | Paraphrases, inferences, ambiguous claims |

The pipeline stops at the first tier that produces a verdict. Most claims in a well-retrieved RAG pipeline never reach Tier 3.

### Large-context models

```python
result = verify(
    claim="...",
    sources=[Source(content=long_document, source_id="contract.pdf")],
    model="gemini-2.0-flash",
    auto_chunk=False,   # pass the full document — no BM25 chunking
)
```

By default, BM25 retrieves the top-k chunks by keyword relevance. A negating clause in a low-scoring chunk — *"that provision was superseded by amendment 3"* — may not reach Tier 3. Set `auto_chunk=False` when your model can fit the full document in context to eliminate this risk.

---

## Domain use cases

**Legal and compliance**
`verify_clause` decomposes contract obligations into atomic verifiable terms. `TermRegistry` pins defined terms as sources — "Effective Date" resolves to its contractual definition before Tier 0 even runs. `STRICT_PROFILE` gives you 0.97 threshold, majority vote, and a full `VerificationAuditRecord` per clause — sufficient for documented compliance chains.

**Financial services**
`verify_analysis` extracts and verifies every numerical claim in a generated report. Tier 2.5 catches unit and magnitude conflicts (300% vs 30%, $4.8M vs $5M) before any LLM call. Every `VERIFIED` result carries a character-offset citation pointing to the source figure — defensible in an audit.

**Clinical and medical**
Set `STRICT_PROFILE`. Each sentence result includes a `VerificationAuditRecord` with the model used, raw LLM response, confidence score, and source ID. Provides a documented verification chain for regulated environments where "the AI said so" is not sufficient.

**Enterprise RAG**
Add `assert_faithful` as a post-generation gate. If the generated answer drifts from the retrieved context, raise before the response reaches the user. Combine with `verify_or_retry` for automatic regeneration. Combine with `max_spend` to cap costs at scale.

**Multi-agent pipelines**
Use `SourceAccumulator` to collect and deduplicate sources across agents. Verify the terminal output against the full source set — not just what the last agent saw:

```python
from groundguard.loaders.accumulator import SourceAccumulator

acc = SourceAccumulator()
acc.add(retrieval_result, provenance="vector_db", agent_id="retriever")
acc.add(web_result, provenance="web_search", agent_id="researcher")

# Verify the final answer against everything retrieved, not just the last hop
result = verify_analysis(final_output, sources=acc.sources(), model="gpt-4o-mini")
```

---

## Local LLMs (no cloud required)

```bash
ollama pull qwen3:14b   # ~9GB Q4_K_M, fits in 15GB RAM
ollama serve
```

```python
result = verify(claim="...", sources=[...], model="ollama/qwen3:14b", max_spend=0.0)
```

Works out of the box — litellm handles the transport. Thinking-mode models (qwen3, DeepSeek-R1, etc.) are fully supported; `<think>` tags are stripped automatically.

---

## When to use groundguard

groundguard works best when:

- Your LLM output is generated from specific documents you control
- Errors have downstream consequences — legal, regulatory, financial, medical
- You're building multi-step pipelines where intermediate hops can introduce drift
- You need an auditable, logged assertion that outputs were verified against sources

It's probably not the right tool when:

- You're building a general-purpose chatbot not tied to specific documents
- You're in early prototyping where accuracy requirements are flexible
- Your use case is creative generation where "grounding" isn't a meaningful constraint

---

## What groundguard is not

- A RAG pipeline or retrieval system — you provide the source documents
- A web scraper or URL fetcher — sources must be developer-provided text
- A hallucination detector for general knowledge — it verifies against *your* sources, not ground truth
- A model evaluation or fine-tuning framework

This distinction matters: groundguard can only tell you whether the output is supported by the documents you provide. If your retrieval step returned the wrong documents, groundguard cannot detect that.

---

## Optional extras

```bash
pip install groundguard[loaders]    # PDF and DOCX ingestion
pip install groundguard[langchain]  # LangChain callback integration
```

```python
# Ingest PDFs and DOCX directly
from groundguard.loaders.helpers import pdf_to_text, docx_to_text

sources = [Source(content=pdf_to_text("contract.pdf"), source_id="contract.pdf")]
result = verify_clause(clause="...", sources=sources)

# LangChain integration
from groundguard.integrations.langchain import AgenticVerifierCallback

callback = AgenticVerifierCallback(model="gpt-4o-mini")
chain.invoke({"query": "What are the termination terms?"}, callbacks=[callback])
# Raises GroundingError if the chain output is not grounded in source_documents
```

---

## FAQ

**How do I verify LLM output against my source documents in Python?**
Install groundguard and call `verify_answer(output, sources, model="gpt-4o-mini")`. It checks whether the generated text is supported by the documents you provide and returns a structured verdict with a citation pointing to the exact supporting excerpt.

**How is this different from asking the LLM to check its own answer?**
The verifying LLM is constrained strictly to your documents — it cannot draw on training knowledge to fill gaps. More importantly, 60–70% of verifications never reach an LLM at all; BM25 resolves them deterministically. When an LLM does run, it shares no context with the generator and uses a different prompt optimized for grounding rather than fluency.

**Does it work with LangChain, OpenAI, Anthropic, Google?**
Yes. groundguard uses litellm under the hood. Any model string that works with litellm works here: `gpt-4o`, `claude-3-5-sonnet-20241022`, `gemini/gemini-2.0-flash`, `ollama/qwen3:14b`, and more. You can mix models across batch items.

**How much does verification cost?**
Most claims are resolved by BM25 for free. When an LLM call runs, the prompt is significantly shorter than a typical generation prompt — only the claim and the top-k relevant chunks are sent, not the full document. Use `estimate_verify_faithfulness_cost()` to get a pre-run estimate before processing at scale.

**What's the difference between groundguard and RAG?**
RAG is a forward pass — retrieve documents, then generate from them. groundguard is a backward pass — verify the generated output against the original documents. They solve different halves of the same problem and are designed to be used together.

**Can it detect hallucinations in general-purpose chatbots?**
No. groundguard verifies output against *your* source documents. It cannot tell you whether a claim is true in the world — only whether it is supported by the documents you provide. For open-world fact-checking, it is not the right tool.

**What happens if the retrieved documents were wrong to begin with?**
groundguard cannot detect that. If your retrieval step returned incorrect or irrelevant documents, verification will succeed against those documents. groundguard guarantees grounding relative to the sources you provide, not ground truth.

---

## License

MIT
