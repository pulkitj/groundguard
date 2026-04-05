# agentic-verifier

LLM-agnostic deterministic assert layer for AI agent outputs. Verifies that generated text is factually grounded in developer-provided source documents.

```
pip install agentic-verifier
```

---

## Quickstart

```python
from agentic_verifier import verify
from agentic_verifier.models.result import Source

sources = [
    Source(content="Q3 revenue was $5 million.", source_id="report.pdf"),
]

result = verify(claim="Revenue was $5M in Q3.", sources=sources, model="gpt-4o-mini")

assert result.is_valid        # True
print(result.status)          # "VERIFIED"
print(result.factual_consistency_score)  # 0.0–1.0
```

---

## What It Is

`agentic-verifier` is a middleware library for high-stakes, document-intensive workflows (legal, compliance, financial, clinical). It sits between your RAG pipeline output and your application, and raises or returns a structured verdict on whether the agent's claim is supported by the source documents you provided.

It is **not**:
- A RAG pipeline builder
- A web scraper or URL fetcher
- A general-purpose agentic framework

---

## How It Works

Every `verify()` call runs a 4-tier pipeline:

| Tier | Module | What it does |
|---|---|---|
| 0 — Classifier | `core/classifier.py` | Splits claim into atomic sentences. Zero cost. |
| 1 — Fuzzy Gate | `tiers/tier1_authenticity.py` | If `agent_provided_evidence` is passed, confirms it appears in source. |
| 2 — BM25 Router | `tiers/tier2_semantic.py` | Routes based on BM25 score: lexical pass (≥0.85) / LLM / escalate-all |
| 3 — LLM Evaluator | `tiers/tier3_evaluation.py` | Sends top-k chunks + claim to LLM. Returns Pydantic-validated verdict. |

Tier 1 is a **gate only** — it never produces a final result. The pipeline always continues to Tier 2 and 3.

---

## Public API

### `verify()`

```python
from agentic_verifier import verify

result = verify(
    claim="Revenue grew by 30% year-over-year.",
    sources=[Source(content="...", source_id="report.pdf")],
    model="gpt-4o-mini",         # any litellm model string
    max_spend=0.50,               # soft USD cap; raises VerificationCostExceededError if exceeded
    auto_chunk=True,              # sliding-window chunk long sources
    tier1_min_similarity=0.90,    # fuzzy gate threshold (0.0–1.0)
    agent_provided_evidence=None, # optional: validate agent-cited quote against source
)
```

### `averify()` — async

```python
result = await averify(claim=..., sources=..., model=...)
```

BM25 (CPU-bound) is dispatched via `asyncio.get_running_loop().run_in_executor()` to avoid blocking the event loop.

### `verify_batch()` / `verify_batch_async()`

```python
from agentic_verifier import verify_batch
from agentic_verifier.models.internal import ClaimInput

inputs = [
    ClaimInput(claim="Claim A", sources=[Source(content="...", source_id="a.pdf")]),
    ClaimInput(claim="Claim B", sources=[Source(content="...", source_id="b.pdf")], model="claude-3-haiku-20240307"),
]

results = verify_batch(inputs=inputs, model="gpt-4o-mini", max_concurrency=5, max_spend=2.0)
```

Each item in the batch shares a `SharedCostTracker`. Items that trigger the spend cap return `status="SKIPPED_DUE_TO_COST"` rather than raising.

> **Jupyter / FastAPI**: `verify_batch()` uses `asyncio.run()` internally and cannot be called from inside a running event loop. Use `await verify_batch_async(...)` directly.

### `verify_structured()`

```python
from pydantic import BaseModel

class RevenueReport(BaseModel):
    revenue_usd: float
    period: str

result = verify_structured(
    claim_dict={"revenue_usd": 5_000_000, "period": "Q3 2024"},
    schema=RevenueReport,
    sources=[Source(content="Q3 2024 revenue was $5 million.", source_id="report.pdf")],
    model="gpt-4o-mini",
)
```

Validates the dict against the Pydantic schema, flattens it to dot-notation (`revenue_usd: 5000000.0\nperiod: Q3 2024`), then calls `verify()`.

---

## VerificationResult

```python
@dataclass
class VerificationResult:
    is_valid: bool
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PARSE_ERROR",
                    "SKIPPED_DUE_TO_COST", "ERROR"]
    overall_verdict: str
    verification_method: str          # "tier2_lexical" | "tier3_llm" | "skipped"
    atomic_claims: list[AtomicClaimResult]
    factual_consistency_score: float  # 0.0–1.0
    sources_used: list[str]           # source_ids that supported the verdict
    rationale: str
    offending_claim: str | None       # first CONTRADICTED atomic claim
    total_cost_usd: float
```

---

## When to Use Which Tier

| Scenario | Tier 2 result | What runs |
|---|---|---|
| Near-verbatim quote from source | `SKIP_LLM_HIGH_CONFIDENCE` | No LLM call (cheap) |
| Paraphrase or inferred claim | `ESCALATE_TO_LLM` | LLM evaluates top-k chunks |
| Completely different vocabulary | `ESCALATE_ALL_LOW_SCORE` | LLM evaluates all chunks (capped at `top_k × 3`) |

---

## `auto_chunk=False` — Large-Context Models

```python
result = verify(
    claim="...",
    sources=[Source(content=long_document, source_id="doc.pdf")],
    model="gemini-1.5-pro",
    auto_chunk=False,   # pass full source to Tier 3 — no chunking
)
```

**When to use**: Large-context models (Gemini 1.5 Pro, Claude 3.5+) that can process full documents in a single call.

**Lost Context Problem**: By default, BM25 retrieves isolated chunks by keyword relevance. A negating statement in a low-scoring chunk (e.g., *"that acquisition was subsequently blocked by the FTC"*) may not be forwarded to Tier 3 — the model never sees it and may return `VERIFIED` incorrectly. Use `auto_chunk=False` for high-stakes verifications where missed negating context is a critical failure mode.

---

## Exception Contract

| Function | Behavior |
|---|---|
| `verify()` / `averify()` | **Fail-loud** — all exceptions propagate except `ParseError` (returned as `status="PARSE_ERROR"`) |
| `verify_batch_async()` | **Fail-contained** — `VerificationCostExceededError` → `SKIPPED_DUE_TO_COST`; all other exceptions → `ERROR` result |

Transient LiteLLM errors (`ServiceUnavailableError`, `RateLimitError`, `APIConnectionError`, `Timeout`) are wrapped as `VerificationFailedError`.

---

## Local LLM (Ollama)

No code changes needed — litellm supports Ollama natively:

```bash
ollama pull qwen3:14b   # ~9GB, fits in 15GB RAM
ollama serve            # starts on http://localhost:11434
```

```python
result = verify(
    claim="...",
    sources=[...],
    model="ollama/qwen3:14b",
    max_spend=0.0,          # local models are free
)
```

Structured output (`response_format`) is supported by Ollama via its OpenAI-compatible endpoint. The `parse_response()` fallback handles any remaining fence-wrapped output.

---

## Optional Extras

```bash
pip install agentic-verifier[loaders]    # PDF and DOCX ingestion
pip install agentic-verifier[langchain]  # LangChain callback integration
pip install agentic-verifier[dev]        # pytest, coverage, asyncio
```

### LangChain Integration

```python
from agentic_verifier.integrations.langchain import AgenticVerifierCallback

callback = AgenticVerifierCallback(model="gpt-4o-mini")
chain.invoke({"query": "What was Q3 revenue?"}, callbacks=[callback])
# Raises VerificationFailedError if the chain output cannot be verified
```

Supported chain types: `RetrievalQA` and LCEL RAG chains that emit `source_documents` and `result` in their output dict.

### PDF / DOCX Loaders

```python
from agentic_verifier.loaders.helpers import pdf_to_text, docx_to_text

text = pdf_to_text("report.pdf")
sources = [Source(content=text, source_id="report.pdf")]
result = verify(claim="...", sources=sources, model="gpt-4o-mini")
```

---

## Out of Scope

- Web scraping / URL fetching (source documents must be developer-provided)
- Training or fine-tuning models
- General-purpose question answering without source documents
- Replacing a RAG retrieval system (provide chunks yourself or use `auto_chunk=True`)

---

## License

MIT
