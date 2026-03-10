# Suffix Automaton + Dynamic-Length Speculative Decoding

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthew-Matta/BasetenResearch_Project/blob/main/notebooks/demo.ipynb)

This implements **dynamic-length speculation** — the explicit future work named in Baseten's [SA MTP blog post (Jan 27, 2026)](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/#suffix-automaton-decoding) — on top of a full reimplementation of the dual-SA speculative decoding architecture from `sa_spec`. Run the Colab notebook to see SA acceptance rates and TPS comparisons on repetitive code generation (temp=1.0, T4 GPU).

---

## What this implements

### 1. Pure-Python Suffix Automaton (`src/suffix_automaton.py`)

O(n) construction via Blumer's algorithm — same algorithm as Baseten's C++/CUDA `sa_spec` repo, reimplemented in Python for portability. Supports:

- Online `extend_one(token)` — O(1) amortized per token
- `query(context, max_draft_len, temperature)` — walks automaton from current context suffix, returns draft candidates + match length; temperature-aware draft selection (greedy uses `last_tok`, sampled uses frequency-weighted selection — picks the most common continuation seen during construction)
- `DualSuffixAutomaton` — static automaton from prompt + dynamic automaton extended token-by-token during generation (mirrors the dual-SA architecture in `sa_spec`)

### 2. Hybrid Speculative Decoding (`src/speculative_decode.py`)

Five modes in a single `HybridSpecDecoder.generate()` call:

| Mode | Draft source | Draft length |
|------|-------------|--------------|
| `autoregressive` | None | 1 |
| `specdec` | Draft model | Fixed |
| `sa_only` | Suffix automaton | Fixed |
| `hybrid_fixed` | SA → draft model fallback | Fixed |
| `hybrid_dynamic` | SA → draft model fallback | **Adaptive** |

Mathematically exact rejection sampling (Leviathan et al. 2023): `accept_prob = min(1, p_target / p_draft)`. For SA drafts, `p_draft = 1` (deterministic), so `accept_prob = p_target(token)`.

KV cache maintained across all decoding steps — each forward pass processes only the new draft tokens, not the full growing sequence.

### 3. Dynamic Length Controller (DLC) — novel contribution

The explicit future work from Baseten's Jan 27 post: *"dynamically adjusting the number of speculative tokens based on observed acceptance rates."*

This implementation extends the idea beyond draft **length** to draft **source routing**:
- Separate rolling-window acceptance rate trackers for SA and draft-model sources
- Draft length adjusts up/down based on 80%/40% thresholds, clamped to [2, 10]
- **Cost-aware source routing**: when `rate × draft_len ≤ 1.0`, the draft model is not economical — DLC falls back to autoregressive, avoiding the overhead trap

The Section 9 scatter plot visualizes this per-step routing: SA (free), draft model (when economical), and AR fallback (when neither source justifies overhead).

---

## Results

*Run the Colab notebook end-to-end on a T4 GPU to reproduce. Numbers vary slightly between runs.*

### SA showcase — the headline result (temperature=1.0)

SA excels when generated tokens repeat substrings from the prompt. The showcase prompt provides two fully-implemented methods and asks the model to complete four more in the same style. At temperature=1.0 the model stochastically revisits patterns like `(self, x: float, y: float) -> float:` and `return x` — exactly what the SA was built from. Frequency-weighted draft selection picks the most common continuation at each state, dramatically improving acceptance over arbitrary token ID selection.

**Why SA drafts are free:** SA proposals are Python dict lookups — zero forward passes. On a memory-bandwidth-bound T4, verifying 4 tokens in a single KV-cached pass costs roughly the same as verifying 1 (the KV cache read dominates). Every accepted SA token is pure profit.

*Re-run Section 9 in the Colab notebook to reproduce.*

### Greedy decoding (temperature=0)

At temperature=0, all five modes produce identical output (verifiable). The `last_tok` strategy ensures the SA proposes exactly the token the model chose last time it saw this context, achieving high acceptance on repeated patterns.

*Re-run Section 10 in the Colab notebook to reproduce.*

### Why draft-model specdec is slower at this model ratio

This is the key insight for understanding speculative decoding economics:

```
With a 3x model ratio (1.5B target / 0.5B draft):
  4 draft tokens at ~0.4x cost each + 1 verification = 2.6 target-equivalent passes
  At ~10% acceptance (temp=1.0): ~1.1 tokens per cycle → 1.1/2.6 = 0.42x (slower than AR)

With a production 10x ratio (e.g. 70B/7B):
  4 draft tokens at ~0.1x cost each + 1 verification = 1.4 target-equivalent passes
  At ~80% acceptance: ~3.4 tokens per cycle → 3.4/1.4 = 2.4x speedup
```

The SA sidesteps this entirely: SA drafts cost 0 forward passes, so even modest acceptance rates yield speedups. The DLC recognizes when draft-model speculation isn't economical (`rate × draft_len ≤ 1.0`) and falls back to autoregressive — avoiding the overhead trap automatically.

---

## Architecture

```
Prompt tokens
     │
     ▼
┌──────────────────────────────┐
│      DualSuffixAutomaton     │
│  prompt_sa (static)          │  ◄── built once from prompt
│  live_sa   (dynamic)         │  ◄── extended per accepted token
└──────────┬───────────────────┘
           │ query(context, draft_len, temperature) → (draft_tokens, match_len)
           ▼
┌──────────────────────────────┐
│    DynamicLengthController   │
│  SA window:    [...]         │  ◄── per-source rolling acceptance
│  draft window: [...]         │
└──────────┬───────────────────┘
           │ get_draft_length(source)
           ▼
┌─────────────────────────────────────────┐
│           HybridSpecDecoder             │
│                                         │   ┌──────────────────┐
│  if SA match ≥ threshold:               │   │  Draft Model     │
│      use SA drafts (p_draft = 1)        │◄──│  Qwen2.5-0.5B   │
│  else:                                  │   └──────────────────┘
│      use draft model tokens             │
│                                         │   ┌──────────────────┐
│  single target forward pass (KV cached) │──►│  Target Model    │
│  exact rejection sampling               │   │  Qwen2.5-1.5B   │
│  update live SA + DLC                   │   └──────────────────┘
└─────────────────────────────────────────┘
```

---

## Quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthew-Matta/BasetenResearch_Project/blob/main/notebooks/demo.ipynb)

Single click — runs on a free T4 GPU. No local setup required.

---

## Connection to Baseten's work

| Baseten's `sa_spec` | This repo |
|---------------------|-----------|
| C++/CUDA SA construction, O(n) Blumer | `SuffixAutomaton` — pure Python, same algorithm |
| Dual automaton (prompt-static + generation-dynamic) | `DualSuffixAutomaton` |
| SA threshold parameter (default 4 in sa_spec) | `sa_threshold` (configurable, default 2) |
| Draft model fallback when SA match insufficient | `HybridSpecDecoder` hybrid modes |
| TPS / TTFT / E2E latency / acceptance rate metrics | `GenerationMetrics`, `MetricsTracker` |
| **"dynamically adjust speculation length"** — named future work, [Jan 27 post](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/#suffix-automaton-decoding) | `DynamicLengthController` — **implemented here** |

---

## Repo structure

```
src/
  suffix_automaton.py   # O(n) Blumer SA, DualSuffixAutomaton
  speculative_decode.py # HybridSpecDecoder, DynamicLengthController
  benchmark.py          # 5-method benchmark harness with CLI
  utils.py              # MetricsTracker, GenerationMetrics, plots
notebooks/
  demo.ipynb            # Colab-ready walkthrough (10 sections)
results/
  benchmarks.json       # Raw metrics (generated at runtime)
  figures/              # TPS bar chart, draft length plot, acceptance breakdown
```

---

## Models

| Role | Model | VRAM (FP16) |
|------|-------|------------|
| Target | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | ~4 GB |
| Draft | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | ~2 GB |

Same model family as Baseten's published benchmarks. Both fit on a free T4 (16 GB).
