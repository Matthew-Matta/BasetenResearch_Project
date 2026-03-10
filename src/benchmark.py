"""
Benchmark harness for suffix-automaton speculative decoding.

Runs all five decoding modes on a shared prompt set and produces:
  - results/benchmarks.json   — raw metrics per prompt per method
  - results/figures/          — TPS bar chart, dynamic draft length, acceptance rates

Models (defaults):
  target: Qwen/Qwen2.5-Coder-1.5B-Instruct
  draft:  Qwen/Qwen2.5-Coder-0.5B-Instruct

Dataset: glaiveai/code_edits_sample (HuggingFace) with hardcoded fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as script from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.speculative_decode import HybridSpecDecoder
from src.utils import GenerationMetrics, plot_benchmark_results, print_summary_table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["autoregressive", "specdec", "sa_only", "hybrid_fixed", "hybrid_dynamic"]

DEFAULT_TARGET = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_DRAFT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

FALLBACK_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list.",
    "Implement a Python class for a max-heap data structure with push and pop methods.",
    "Write a Python decorator that caches function results with a TTL (time-to-live).",
    "Implement quicksort in Python with in-place partitioning.",
    "Write a Python context manager for timing code blocks.",
    "Implement a simple LRU cache using OrderedDict in Python.",
    "Write a Python function to flatten a nested list of arbitrary depth.",
    "Implement a basic tokenizer that splits text into words and punctuation.",
    "Write a Python generator that yields Fibonacci numbers indefinitely.",
    "Implement a thread-safe counter class using Python's threading module.",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(n_samples: int = 50) -> list[str]:
    """Load prompts from HuggingFace or fall back to hardcoded examples."""
    try:
        from datasets import load_dataset as hf_load
        print("Loading glaiveai/code_edits_sample from HuggingFace...")
        ds = hf_load("glaiveai/code_edits_sample", split="train")
        prompts = []
        for item in ds:
            # Extract instruction/prompt field (varies by dataset schema)
            text = item.get("instruction") or item.get("prompt") or item.get("input") or ""
            if isinstance(text, str) and len(text.strip()) > 20:
                prompts.append(text.strip())
            if len(prompts) >= n_samples:
                break
        if prompts:
            print(f"Loaded {len(prompts)} prompts from HuggingFace dataset.")
            return prompts
    except Exception as e:
        print(f"HuggingFace dataset unavailable ({e}), using fallback prompts.")

    # Cycle fallback prompts to reach n_samples
    prompts = []
    while len(prompts) < n_samples:
        prompts.extend(FALLBACK_PROMPTS)
    return prompts[:n_samples]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    target_name: str = DEFAULT_TARGET,
    draft_name: str = DEFAULT_DRAFT,
    device: str | None = None,
) -> tuple:
    """Load target and draft models with FP16 if on CUDA."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading target model: {target_name}")
    target_tok = AutoTokenizer.from_pretrained(target_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map="auto"
    )

    print(f"Loading draft model: {draft_name}")
    draft_tok = AutoTokenizer.from_pretrained(draft_name)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=dtype, device_map="auto"
    )

    return target_model, target_tok, draft_model, draft_tok, device


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    decoder: HybridSpecDecoder,
    prompts: list[str],
    method: str,
    max_new_tokens: int = 100,
    num_draft_tokens: int = 4,
    sa_threshold: int = 2,
    temperature: float = 0.0,
    verbose: bool = False,
) -> list[GenerationMetrics]:
    """Run *method* on all *prompts* and return per-prompt metrics."""
    from tqdm import tqdm

    # Warmup CUDA kernels before timing (idempotent if already warmed)
    decoder.warmup()

    results = []
    for i, prompt in enumerate(tqdm(prompts, desc=method, leave=False)):
        try:
            _, metrics = decoder.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode=method,
                num_draft_tokens=num_draft_tokens,
                sa_threshold=sa_threshold,
                temperature=temperature,
            )
            results.append(metrics)
            if verbose:
                print(
                    f"  [{i+1}/{len(prompts)}] TPS={metrics.tokens_per_second:.1f} "
                    f"acc={metrics.acceptance_rate:.2f} src={metrics.source_history[:3]}"
                )
        except Exception as e:
            print(f"  Error on prompt {i}: {e}")
    return results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(all_results: dict[str, list[GenerationMetrics]], path: str = "results/benchmarks.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialisable = {
        method: [m.to_dict() for m in metrics_list]
        for method, metrics_list in all_results.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative decoding benchmark")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--sa-threshold", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load data
    prompts = load_dataset(n_samples=args.n_samples)
    print(f"Benchmark: {len(prompts)} prompts, methods: {args.methods}")

    # Load models
    target_model, target_tok, draft_model, draft_tok, device = load_models(
        target_name=args.target,
        draft_name=args.draft,
        device=args.device,
    )

    decoder = HybridSpecDecoder(
        target_model=target_model,
        target_tokenizer=target_tok,
        draft_model=draft_model,
        draft_tokenizer=draft_tok,
        device=device,
    )

    # Run benchmarks
    all_results: dict[str, list[GenerationMetrics]] = {}
    for method in args.methods:
        print(f"\n--- Running: {method} ---")
        metrics_list = run_benchmark(
            decoder=decoder,
            prompts=prompts,
            method=method,
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            sa_threshold=args.sa_threshold,
            verbose=args.verbose,
        )
        all_results[method] = metrics_list

    # Save + visualise
    save_results(all_results, path=os.path.join(args.results_dir, "benchmarks.json"))
    plot_benchmark_results(all_results, output_dir=os.path.join(args.results_dir, "figures"))
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
