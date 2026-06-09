# groundguard RAGTruth Benchmark

Measures groundguard's hallucination-detection accuracy on the
[RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) QA subset
and compares it against a RAGAS faithfulness baseline.

## Dataset

**RAGTruth** (ACL 2024) — real outputs from production RAG systems (GPT-4,
LLaMA-2, Mistral), not synthetic pairs. The QA subset contains 989 samples.

Why RAGTruth over alternatives:

- Real outputs from actual RAG pipelines match groundguard's exact use case
- Word-level hallucination annotations enable stratified results by type
- ACL 2024 publication reduces training-data contamination risk

**Field mapping:**

| RAGTruth field                  | groundguard input               |
| ------------------------------- | ------------------------------- |
| `context`                       | `Source(content=...)`           |
| `output`                        | claim passed to `verify()`      |
| `query`                         | `context=` (task context)       |
| `hallucination_labels_processed`| ground truth label              |

**Binary label derivation:**

```text
hallucinated = evident_conflict == 1  OR  baseless_info == 1
faithful     = evident_conflict == 0  AND baseless_info == 0
```

## Aggregation rule

groundguard returns per-claim verdicts. The response-level verdict is:

```text
hallucinated  if ANY atomic claim is CONTRADICTED or UNVERIFIABLE
faithful      if ALL atomic claims are VERIFIED
```

`UNVERIFIABLE` maps to hallucinated because RAGTruth's `baseless_info` class
(facts absent from the context) is exactly what groundguard surfaces as
`UNVERIFIABLE`. Treating it as faithful would miss all baseless hallucinations.

## RAGAS comparison

RAGAS faithfulness runs once on the same 989 samples and serves as the fixed
baseline for all groundguard phases. Important caveat: **the two tools aggregate
differently**. groundguard uses ANY-claim-fails → hallucinated; RAGAS averages
faithfulness scores across claims and binarises at 0.5. A 20-claim response with
one wrong claim: groundguard flags it hallucinated; RAGAS likely scores it
0.95 → faithful. The comparison measures prompt strategy under the same LLM, not
quality in isolation.

## Setup

### Install dependencies

```bash
pip install groundguard
pip install -r benchmarks/requirements.txt
```

### Set your API key

The benchmark uses [litellm](https://docs.litellm.ai/docs/providers) under the
hood, so any model litellm supports will work — set the appropriate environment
variable for your provider and pass the model string via `--model`.

For example, using Google AI Studio (free tier):

```bash
export GEMINI_API_KEY=your_key_here
# then pass: --model gemini/gemini-3.1-flash-lite
```

## Running

```bash
# 200-sample run with Gemini Flash Lite (AI Studio, free tier)
python benchmarks/run_benchmark.py \
  --phase 1 \
  --sample-size 200 \
  --model gemini/gemini-3.1-flash-lite \
  --output benchmarks/results/my_run.jsonl

# Full 900-sample run with Vertex AI Gemini 2.5 Flash
python benchmarks/run_benchmark.py \
  --phase 1 \
  --model vertex_ai/gemini-2.5-flash \
  --output benchmarks/results/my_run.jsonl

# Use the pip-installed groundguard package instead of repo source
python benchmarks/run_benchmark.py \
  --phase 1 \
  --use-installed \
  --model gemini/gemini-3.1-flash-lite \
  --output benchmarks/results/my_run.jsonl
```

### Options

| Option             | Default                      | Description                              |
| ------------------ | ---------------------------- | ---------------------------------------- |
| `--phase`          | required                     | Benchmark phase (see Phases below)       |
| `--sample-size`    | 900                          | Number of samples to evaluate            |
| `--model`          | `vertex_ai/gemini-2.5-flash` | Any litellm model string                 |
| `--output`         | auto-named by phase          | Path to JSONL output file                |
| `--no-auto-chunk`  | —                            | Pass full source text to LLM             |
| `--profile strict` | —                            | Higher BM25 threshold, more top-k chunks |
| `--use-installed`  | —                            | Use pip-installed groundguard            |

### Phases

| Phase | What it tests                                                         |
| ----- | --------------------------------------------------------------------- |
| `1`   | Baseline: GENERAL_PROFILE, `auto_chunk=True`                          |
| `3a`  | Fix A: BM25 threshold raised `0.85 → 0.92`, routes more claims to LLM |
| `3b`  | Fix B: `auto_chunk=False` — full source text bypasses BM25 chunking   |
| `3c`  | Profile: STRICT_PROFILE — demonstrates precision/recall tradeoff      |

RAGAS runs once and is reused as the fixed baseline across all phases.

### Running the RAGAS baseline

```bash
python benchmarks/run_benchmark.py \
  --run-ragas \
  --model vertex_ai/gemini-2.5-flash \
  --output benchmarks/results/ragas_baseline.jsonl
```

## Output format

Results are written as newline-delimited JSON (JSONL). Each line is one sample:

```json
{
  "sample_id": "12345",
  "ground_truth": "hallucinated",
  "groundguard_status": "CONTRADICTED",
  "has_contradiction": true,
  "unverifiable_ratio": 0.25,
  "contradicted_claim_count": 1,
  "tier3_call_count": 1,
  "latency_ms": 8200,
  "cost_usd": 0.000042,
  "phase": "1",
  "auto_chunk": true,
  "model": "gemini/gemini-3.1-flash-lite"
}
```

Possible `groundguard_status` values:
`VERIFIED`, `CONTRADICTED`, `UNVERIFIABLE`, `PARSE_ERROR`, `INVARIANT_ERROR`, `ERROR`.

## Resume

The script checkpoints every sample. If a run is interrupted, re-run the same
command with the same `--output` file — already-processed samples are skipped.
Transient infrastructure failures (503 exhaustion) are not checkpointed and will
be retried automatically on resume.
