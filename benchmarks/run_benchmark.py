import argparse
import json
import os
import sys
import time
import traceback

# Configure stdout/stderr to use utf-8 encoding to avoid Windows console errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

from datasets import Dataset, load_dataset

# Ensure groundguard is importable from the root directory.
# Pass --use-installed to skip this and use the venv-installed package instead.
if "--use-installed" not in sys.argv:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Monkeypatch missing legacy langchain_community module for RAGAS compatibility
try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    pass

from groundguard.core.verifier import verify
from groundguard.exceptions import InvariantError, VerificationFailedError
from groundguard.models.result import Source
from groundguard.profiles import GENERAL_PROFILE, STRICT_PROFILE

# Configure Google Vertex AI environment variables for the benchmark
project_id = os.environ.get("VERTEXAI_PROJECT", os.environ.get("VERTEX_PROJECT", "groundguard-benchmark"))
location = os.environ.get("VERTEXAI_LOCATION", os.environ.get("VERTEX_LOCATION", "us-central1"))

os.environ["VERTEXAI_PROJECT"] = project_id
os.environ["VERTEX_PROJECT"] = project_id
os.environ["VERTEXAI_LOCATION"] = location
os.environ["VERTEX_LOCATION"] = location

def load_qa_dataset(sample_size):
    print("Loading wandb/RAGTruth-processed dataset...")
    ds = load_dataset("wandb/RAGTruth-processed")
    qa_test = ds["test"].filter(lambda x: x["task_type"] == "QA")
    
    # Slice the dataset according to sample_size
    sliced_ds = qa_test.select(range(min(sample_size, len(qa_test))))
    print(f"Loaded {len(sliced_ds)} QA test samples.")
    return sliced_ds

def run_groundguard_eval(samples, phase, model, auto_chunk_override, profile_override, output_file):
    # Set up configuration based on phase
    # Phase 1: v0.1.3, GENERAL_PROFILE, auto_chunk=True
    # Phase 3a: v0.1.4 (threshold raised to 0.92 in profiles.py), GENERAL_PROFILE, auto_chunk=True
    # Phase 3b: v0.1.3, GENERAL_PROFILE, auto_chunk=False
    # Phase 3c: v0.1.3, STRICT_PROFILE, auto_chunk=True
    
    from dataclasses import replace
    profile = GENERAL_PROFILE
    if phase == "3a":
        profile = replace(GENERAL_PROFILE, tier2_lexical_threshold=0.92)
        print("Using GENERAL_PROFILE with tier2_lexical_threshold=0.92")
    elif phase == "3c" or profile_override == "strict":
        profile = STRICT_PROFILE
        print("Using STRICT_PROFILE")
    else:
        print("Using GENERAL_PROFILE")
        
    auto_chunk = True
    if phase == "3b":
        auto_chunk = False
    if auto_chunk_override is not None:
        auto_chunk = auto_chunk_override
        
    print(f"Auto-chunk configuration: {auto_chunk}")
    
    # Read existing checkpoints to support resume
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["sample_id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_ids)} already processed samples. Resuming...")
        
    # Open output file in append mode
    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples):
            sample_id = sample["id"]
            if sample_id in processed_ids:
                continue
                
            query = sample["query"]
            context = sample["context"]
            output = sample["output"]
            gt_processed = sample["hallucination_labels_processed"]
            
            # Ground truth label mapping:
            # hallucinated = evident_conflict == 1 OR baseless_info == 1
            gt_hallucinated = (gt_processed["evident_conflict"] == 1 or gt_processed["baseless_info"] == 1)
            gt_label = "hallucinated" if gt_hallucinated else "faithful"
            
            print(f"[{idx+1}/{len(samples)}] Verifying sample {sample_id} ({gt_label})...")
            
            start_time = time.time()
            try:
                # Core groundguard verifier call (sequential evaluation)
                result = verify(
                    claim=output,
                    sources=[Source(content=context, source_id="ragtruth_context")],
                    context=query,
                    model=model,
                    profile=profile,
                    auto_chunk=auto_chunk
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Extract routing metric counts
                unverifiable_count = sum(1 for a in result.atomic_claims if a.status == "UNVERIFIABLE")
                contradicted_count = sum(1 for a in result.atomic_claims if a.status == "CONTRADICTED")
                
                total_claims = len(result.atomic_claims)
                has_contradiction = contradicted_count > 0
                unverifiable_ratio = (unverifiable_count / total_claims) if total_claims > 0 else 0.0

                # In groundguard verify() pipeline, methods are tracked overall
                tier3_calls = 1 if result.verification_method == "tier3_llm" else 0
                branch_a_bypasses = 1 if result.verification_method == "tier2_lexical" else 0
                tier25_intercepts = 1 if result.verification_method == "tier25_numerical" else 0
                
                record = {
                    "sample_id": sample_id,
                    "ground_truth": gt_label,
                    "groundguard_status": result.status,
                    "has_contradiction": has_contradiction,
                    "unverifiable_ratio": unverifiable_ratio,
                    "unverifiable_claim_count": unverifiable_count,
                    "contradicted_claim_count": contradicted_count,
                    "tier3_call_count": tier3_calls,
                    "branch_a_bypass_count": branch_a_bypasses,
                    "tier25_intercept_count": tier25_intercepts,
                    "total_claim_count": total_claims,
                    "factual_consistency_score": result.factual_consistency_score,
                    "source_count": 1,
                    "latency_ms": latency_ms,
                    "cost_usd": result.total_cost_usd,
                    "phase": phase,
                    "auto_chunk": auto_chunk,
                    "model": model
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"   GG Status: {result.status} | Contradicted: {has_contradiction} | Unverifiable Ratio: {unverifiable_ratio:.2f} | Latency: {latency_ms}ms | Cost: ${result.total_cost_usd:.6f}")
                
            except InvariantError as e:
                latency_ms = int((time.time() - start_time) * 1000)
                cost = getattr(e, "cost_usd", 0.0)
                print(f"   InvariantError on sample {sample_id}: {e} | Latency: {latency_ms}ms | Cost: ${cost:.6f}")
                record = {
                    "sample_id": sample_id,
                    "ground_truth": gt_label,
                    "groundguard_status": "INVARIANT_ERROR",
                    "error": str(e),
                    "has_contradiction": False,
                    "unverifiable_ratio": None,
                    "unverifiable_claim_count": None,
                    "contradicted_count": None,
                    "total_claim_count": None,
                    "factual_consistency_score": None,
                    "source_count": 1,
                    "tier3_call_count": None,
                    "branch_a_bypass_count": None,
                    "tier25_intercept_count": None,
                    "latency_ms": latency_ms,
                    "cost_usd": cost,
                    "phase": phase,
                    "auto_chunk": auto_chunk,
                    "model": model
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            except VerificationFailedError as e:
                # Transient infrastructure failure (503 exhaustion, connection error).
                # Do NOT write a checkpoint — sample remains eligible for retry on resume.
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"   Transient failure on sample {sample_id} (will retry on resume): {e} | Latency: {latency_ms}ms")
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"   GG Failure on sample {sample_id}: {e} | Latency: {latency_ms}ms")
                traceback.print_exc()
                record = {
                    "sample_id": sample_id,
                    "ground_truth": gt_label,
                    "groundguard_status": "ERROR",
                    "error": str(e),
                    "has_contradiction": False,
                    "unverifiable_ratio": None,
                    "unverifiable_claim_count": None,
                    "contradicted_count": None,
                    "total_claim_count": None,
                    "factual_consistency_score": None,
                    "source_count": 1,
                    "tier3_call_count": None,
                    "branch_a_bypass_count": None,
                    "tier25_intercept_count": None,
                    "latency_ms": latency_ms,
                    "cost_usd": 0.0,
                    "phase": phase,
                    "auto_chunk": auto_chunk,
                    "model": model
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                
            # Minimal sleep — Flash Lite call latency (~60-130s/sample) is the binding constraint, not RPM
            time.sleep(2)

def run_ragas_eval(samples, model, output_file, ragas_threshold):
    print("Initializing RAGAS ChatVertexAI wrapper...")
    import litellm
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_google_vertexai import ChatVertexAI
    from ragas import evaluate
    from ragas.metrics import faithfulness

    class _TokenTracker(BaseCallbackHandler):
        """Accumulates token counts across all LLM sub-calls during one RAGAS evaluate()."""

        def __init__(self):
            self.input_tokens = 0
            self.output_tokens = 0

        def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = gen.generation_info or {}
                    usage = info.get("usage_metadata") or {}
                    self.input_tokens += usage.get("prompt_token_count", 0)
                    self.output_tokens += usage.get("candidates_token_count", 0)

        def cost_usd(self, ragas_model: str) -> float | None:
            if self.input_tokens == 0 and self.output_tokens == 0:
                return None
            try:
                in_cost, out_cost = litellm.cost_per_token(
                    model=ragas_model,
                    prompt_tokens=self.input_tokens,
                    completion_tokens=self.output_tokens,
                )
                return in_cost + out_cost
            except Exception:
                return None
    
    # Ragas expects a Vertex AI model wrapper
    # Map the model string to langchain_google_vertexai parameters
    # vertex_ai/gemini-2.5-flash -> model_name="gemini-2.5-flash"
    model_name = model.split("/")[-1] if "/" in model else model
    
    chat_model = ChatVertexAI(
        model_name=model_name,
        project=os.environ.get("VERTEXAI_PROJECT", "groundguard-benchmark"),
        location=os.environ.get("VERTEXAI_LOCATION", "us-central1"),
        temperature=0.0
    )
    
    # Read existing checkpoints to support resume
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["sample_id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_ids)} already processed RAGAS samples. Resuming...")
        
    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples):
            sample_id = sample["id"]
            if sample_id in processed_ids:
                continue
                
            query = sample["query"]
            context = sample["context"]
            output = sample["output"]
            gt_processed = sample["hallucination_labels_processed"]
            
            # Ground truth label
            gt_hallucinated = (gt_processed["evident_conflict"] == 1 or gt_processed["baseless_info"] == 1)
            gt_label = "hallucinated" if gt_hallucinated else "faithful"
            
            print(f"[{idx+1}/{len(samples)}] Evaluating RAGAS for sample {sample_id} ({gt_label})...")
            
            # Format inputs as a single-item Dataset
            data_samples = {
                'question': [query],
                'answer': [output],
                'contexts': [[context]], # Context is a list of lists in RAGAS
            }
            dataset = Dataset.from_dict(data_samples)
            
            tracker = _TokenTracker()
            start_time = time.time()
            try:
                # Run Ragas evaluation — attach tracker to capture per-call token usage
                results = evaluate(
                    dataset,
                    metrics=[faithfulness],
                    llm=chat_model,
                    callbacks=[tracker],
                )
                latency_ms = int((time.time() - start_time) * 1000)
                import math
                scores = results["faithfulness"]
                raw_score = scores[0] if scores else float("nan")
                ragas_score = 0.0 if math.isnan(raw_score) else raw_score
                ragas_verdict = "hallucinated" if ragas_score < ragas_threshold else "faithful"
                cost = tracker.cost_usd(model)

                record = {
                    "sample_id": sample_id,
                    "ground_truth": gt_label,
                    "ragas_score": ragas_score,
                    "ragas_verdict": ragas_verdict,
                    "latency_ms": latency_ms,
                    "cost_usd": cost,
                    "ragas_input_tokens": tracker.input_tokens,
                    "ragas_output_tokens": tracker.output_tokens,
                    "model": model
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                cost_str = f"${cost:.6f}" if cost is not None else "unknown"
                print(f"   RAGAS Score: {ragas_score} ({ragas_verdict}) | Latency: {latency_ms}ms | Cost: {cost_str} ({tracker.input_tokens}in/{tracker.output_tokens}out tokens)")
                
            except Exception as e:
                print(f"   RAGAS Failure on sample {sample_id}: {e}")
                traceback.print_exc()
                # Write error record
                record = {
                    "sample_id": sample_id,
                    "ground_truth": gt_label,
                    "ragas_score": -1.0,
                    "error": str(e),
                    "model": model
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                
            # Sleep to manage rate limit pressure
            time.sleep(1.0)

def main():
    parser = argparse.ArgumentParser(description="groundguard RAGTruth Benchmark Suite")
    parser.add_argument("--phase", type=str, choices=["1", "3a", "3b", "3c"], help="groundguard benchmark phase")
    parser.add_argument("--sample-size", type=int, default=900, help="number of samples to evaluate (default: 900)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. vertex_ai/gemini-2.5-flash)")
    parser.add_argument("--auto-chunk", dest="auto_chunk", action="store_true", help="force enable auto_chunk")
    parser.add_argument("--no-auto-chunk", dest="auto_chunk", action="store_false", help="force disable auto_chunk")
    parser.set_defaults(auto_chunk=None)
    parser.add_argument("--profile", type=str, choices=["general", "strict"], help="profile override")
    parser.add_argument("--run-ragas", action="store_true", help="run RAGAS baseline instead of groundguard")
    parser.add_argument("--ragas-threshold", type=float, default=0.5, help="threshold to binarize RAGAS score")
    parser.add_argument("--output", type=str, default=None, help="override output file path")
    parser.add_argument("--use-installed", action="store_true", help="use venv-installed groundguard instead of repo source")

    args = parser.parse_args()

    if not args.run_ragas and not args.phase:
        parser.error("Either --phase or --run-ragas is required.")

    # Ensure results directory exists
    os.makedirs("benchmarks/results", exist_ok=True)

    # Load dataset
    samples = load_qa_dataset(args.sample_size)

    if args.run_ragas:
        output_file = args.output or "benchmarks/results/ragas_baseline.jsonl"
        print(f"Starting RAGAS baseline run on {len(samples)} samples.")
        print(f"Threshold: {args.ragas_threshold}")
        print(f"Output file: {output_file}")
        run_ragas_eval(samples, args.model, output_file, args.ragas_threshold)
    else:
        # Determine output file name based on phase
        # Phase 1: phase1_v013.jsonl
        # Phase 3a: phase3a_v014.jsonl
        # Phase 3b: phase3b_v013_nochunk.jsonl
        # Phase 3c: phase3c_v013_strict.jsonl
        if args.output:
            output_file = args.output
        elif args.phase == "1":
            output_file = "benchmarks/results/phase1_v013.jsonl"
        elif args.phase == "3a":
            output_file = "benchmarks/results/phase3a_v014.jsonl"
        elif args.phase == "3b":
            output_file = "benchmarks/results/phase3b_v013_nochunk.jsonl"
        elif args.phase == "3c":
            output_file = "benchmarks/results/phase3c_v013_strict.jsonl"

        print(f"Starting groundguard Phase {args.phase} run on {len(samples)} samples.")
        print(f"Output file: {output_file}")
        run_groundguard_eval(
            samples=samples,
            phase=args.phase,
            model=args.model,
            auto_chunk_override=args.auto_chunk,
            profile_override=args.profile,
            output_file=output_file
        )

if __name__ == "__main__":
    main()
