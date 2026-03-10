# Suffix Automaton + Dynamic-Length Speculative Decoding

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthew-Matta/BasetenResearch_Project/blob/main/notebooks/demo.ipynb)

**2.16x throughput vs autoregressive** on repetitive code generation (49.2 vs 22.8 TPS, T4 GPU) using zero-cost suffix automaton drafts with suffix-link fallback.

Implements **dynamic-length speculation**, future work named in Baseten's [SA MTP blog post (Jan 27, 2026)](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/#suffix-automaton-decoding) — on top of a full reimplementation of the dual-SA speculative decoding architecture from `sa_spec`, extended with per-source draft routing (SA vs draft model vs AR fallback).

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

This implementation extends the idea beyond draft **length** to draft **source routing**:
- Separate rolling-window acceptance rate trackers for SA and draft-model sources
- Draft length adjusts up/down based on 80%/40% thresholds, clamped to [2, 10]
- **Cost-aware source routing**: when `rate × draft_len ≤ 1.0`, the draft model is not economical — DLC falls back to autoregressive, avoiding the overhead trap

The Section 9 scatter plot visualizes this per-step routing: SA (free), draft model (when economical), and AR fallback (when neither source justifies overhead).

---

## Results (T4 GPU, Colab free tier)

*All numbers from a single end-to-end notebook run. Re-run the Colab to reproduce — numbers vary slightly between runs.*

### SA showcase — repetitive code prompt (temperature=1.0)

The showcase prompt provides two fully-implemented Calculator methods and asks the model to complete four more in the same style. At temperature=1.0 the model stochastically revisits patterns like `(self, x: float, y: float) -> float:` and `return x`. The substrings the SA was built from.

| Mode | TPS | SA acceptance | Avg draft len | vs AR |
|------|-----|---------------|---------------|-------|
| `autoregressive` | 22.8 | — | 1.00 | 1.00x |
| `sa_only` | **49.2** | **38.7%** | **6.28** | **2.16x** |
| `hybrid_dynamic` | 16.7 | 23.9% | 4.39 | 0.73x |

**sa_only at 2.16x** is the headline: suffix-link fallback produces ~6 draft tokens per SA firing (up from ~1.3 before the fix), and every accepted token is pure profit since SA proposals are dict lookups — zero forward passes.

`hybrid_dynamic` is slower here because the DLC occasionally routes to the draft model, which is net negative at this 3x model ratio. On production hardware with a 10x+ ratio, the hybrid mode would benefit from both sources.

### Greedy SA showcase — SA's best case (temperature=0)

At temperature=0, SA's `last_tok` prediction is deterministic: it records what the target model chose last time it saw this context. On repeated patterns, this matches the target's argmax perfectly.

| Mode | TPS | Acceptance | Avg draft len | SA acceptance | vs AR |
|------|-----|------------|---------------|---------------|-------|
| `autoregressive` | 23.0 | 100.0% | 1.00 | — | 1.00x |
| `specdec` | 12.1 | 55.9% | 3.94 | — | 0.53x |
| `sa_only` | **46.6** | 44.1% | **5.85** | **39.5%** | **2.03x** |
| `hybrid_dynamic` | **31.1** | 43.7% | 5.89 | 39.5% | **1.35x** |

### Mini-benchmark — 10 generic code prompts (temperature=0)

On diverse, non-repetitive prompts (glaiveai/code_edits_sample), SA fires less often. This is the baseline — SA helps on repetitive patterns, breaks even elsewhere.

| Mode | TPS | Acceptance | SA acc | Draft acc | Avg draft len |
|------|-----|------------|--------|-----------|---------------|
| `autoregressive` | 22.7 | 100.0% | — | — | 1.00 |
| `specdec` | 7.7 | 27.2% | — | 27.2% | 3.91 |
| `sa_only` | 22.7 | 60.4% | 8.8% | — | 1.64 |
| `hybrid_fixed` | 8.5 | 25.8% | 9.5% | 29.6% | 4.34 |
| `hybrid_dynamic` | 12.0 | 53.7% | 8.6% | 24.7% | 1.96 |

SA breaks even on generic prompts (no downside — zero-cost proposals), while the DLC in `hybrid_dynamic` correctly routes away from the draft model when it's not economical (12.0 vs 8.5 TPS for `hybrid_fixed`).

### Why draft-model specdec is slower at this model ratio

With a 3x ratio (1.5B/0.5B), the draft model costs ~0.4x per token — too expensive for its acceptance rate:

```
3x ratio (this repo):   4 drafts × 0.4x + 1 verify = 2.6x cost → 1.1 tokens/cycle → 0.42x (slower)
10x ratio (production):  4 drafts × 0.1x + 1 verify = 1.4x cost → 3.4 tokens/cycle → 2.4x speedup
```

SA sidesteps this entirely: 0 forward passes per draft token. The DLC recognizes when draft-model speculation isn't economical (`rate × draft_len ≤ 1.0`) and falls back to AR automatically — visible in the hybrid_dynamic source routing scatter plot (Section 9 of the notebook).

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
