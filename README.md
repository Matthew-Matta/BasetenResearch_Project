# Suffix Automaton + Dynamic-Length Speculative Decoding

A weekend implementation of suffix-automaton speculative decoding with **dynamic draft length control** — the explicit future work item named in Baseten's [SA MTP blog post (Jan 27, 2026)](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/#suffix-automaton-decoding) and left unshipped in their Part 2 benchmarking post (Feb 5, 2026). SA showcase achieves **1.26x TPS** on repetitive code generation (temp=1.0, T4 GPU).

Built to understand and extend Baseten's [`sa_spec`](https://github.com/basetenlabs/sa_spec) architecture. All five decoding modes run on a free Colab T4.

---

## What this implements

### 1. Pure-Python Suffix Automaton (`src/suffix_automaton.py`)

O(n) construction via Blumer's algorithm — same algorithm as Baseten's C++/CUDA `sa_spec` repo, reimplemented in Python for portability. Supports:

- Online `extend_one(token)` — O(1) amortized per token
- `query(context, max_draft_len)` — walks automaton from current context suffix, returns draft candidates + match length
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

### 3. Dynamic Draft Length Controller — novel contribution

The explicit future work from Baseten's Jan 27 post: *"dynamically adjusting the number of speculative tokens based on observed acceptance rates."*

Separate rolling-window acceptance rate trackers for SA and draft-model sources. Draft length adjusts up/down based on 80%/40% thresholds, clamped to [2, 10].

```python
# SA and draft model tracked independently
# High acceptance (>80%) → increase draft_len by 1
# Low acceptance  (<40%) → decrease draft_len by 1
```

---

## Results

All numbers measured on Google Colab T4 (free tier), Qwen2.5-Coder 1.5B/0.5B.

### SA showcase — **the novel contribution** (temperature=1.0)

SA excels when generated tokens repeat substrings from the prompt. The showcase prompt provides two fully-implemented methods and asks the model to complete four more in the same style. The SA is built from the prompt and contains repeated substrings like `(self, x: float, y: float) -> float:`, `"""`, and `return x`. At temperature=1.0 the model sometimes regenerates these patterns verbatim — giving the SA real match opportunities.

| Mode | TPS | SA acc % | Avg draft len | Speedup |
|------|-----|----------|---------------|---------|
| autoregressive | 23.5 | — | 1.0 | 1x |
| sa_only | **29.7** | 72% | ~1.4 | **1.26x** |
| hybrid_dynamic | ~27 | ~60% | adaptive | ~1.15x |

*Numbers from a single Colab T4 run — re-run the SA showcase cell (Section 9) to reproduce.*

### Greedy decoding (temperature=0) — correct output, honest speedup

At temperature=0, all five modes produce identical output (verifiable). Whether speculative decoding beats autoregressive depends on model-family alignment. With this 3x ratio (1.5B/0.5B):

| Mode | TPS | Acceptance | Avg draft len | Speedup |
|------|-----|-----------|---------------|---------|
| autoregressive | ~23 | — | 1.0 | 1x |
| specdec | ~25–74 | 14–80% | 4.0 | varies |
| hybrid_dynamic | similar | similar | adaptive | varies |

**Re-run the greedy cell (Section 10) to see your machine's actual numbers** — the TPS output is now in a separate cell so it's visible without scrolling past charts.

**Why results vary:** speculative decoding speedup at greedy depends on prompt/model alignment. Baseten's production deployments use a 70B/7B ratio (10x); this repo uses 1.5B/0.5B (3x), so acceptance rates are lower and overhead can dominate on some prompts.

### Sampled decoding — mini-benchmark (temperature=1.0, 10 prompts)

| Mode | TPS | Acceptance | Notes |
|------|-----|-----------|-------|
| autoregressive | ~22 | 100% | Baseline |
| specdec | ~5–7 | 10–27% | 3x model ratio: draft overhead dominates |
| sa_only | ~22 | 100% | SA overhead negligible |
| hybrid_fixed | ~5–7 | 10–27% | Same bottleneck as specdec |
| hybrid_dynamic | ~5–7 | 10–27% | Same bottleneck as specdec |

**Why specdec is slower at temperature=1.0 with this model size:** At 3x model ratio (1.5B/0.5B), acceptance rates are too low to offset draft+verification overhead. Baseten's production deployments use 70B/7B (10x) where gains are 2-3x TPS. This is expected behavior — the SA showcase above shows the real win.

---

## When does SA fire?

SA finds matches when generated tokens repeat substrings of the **recent context** (a sliding window of the last ~30 tokens). This happens with:

- **Repetitive code** — multiple methods with shared signatures, boilerplate
- **Long context** — more tokens in the live automaton = more match opportunities
- **Temperature=1.0** — model stochastically revisits prompt patterns; greedy (temp=0) often takes a deterministic path that diverges from prompt phrasing
- **Prompt that contains the patterns** — SA is built from prompt tokens, so the prompt must contain substrings that the model will regenerate

SA does NOT help with novel code generation from a short prompt — tokens are mostly new content. This matches `sa_spec`'s finding that SA acceptance rates improve with longer generation and higher context repetition.

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
           │ query(context, draft_len) → (draft_tokens, match_len)
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

### Google Colab (T4 GPU — free tier)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthew-Matta/BasetenResearch_Project/blob/main/notebooks/demo.ipynb)

### Local

```bash
pip install -r requirements.txt

# Full benchmark (GPU recommended)
python -m src.benchmark --n-samples 10 --max-new-tokens 80

# SA unit tests only (CPU, no models needed)
python -c "
from src.suffix_automaton import SuffixAutomaton
sa = SuffixAutomaton()
sa.build([1, 2, 3, 1, 2])
drafts, match_len = sa.query([1, 2], max_draft_len=4)
assert match_len == 2 and 3 in drafts
print('SA unit test passed')
"
```

---

## Repo structure

```
src/
  suffix_automaton.py   # O(n) Blumer SA, DualSuffixAutomaton
  speculative_decode.py # HybridSpecDecoder, DynamicLengthController
  benchmark.py          # 5-method benchmark harness with CLI
  utils.py              # MetricsTracker, GenerationMetrics, plots
notebooks/
  demo.ipynb            # Colab-ready walkthrough (8 sections)
results/
  benchmarks.json       # Raw metrics (generated at runtime)
  figures/              # TPS bar chart, draft length plot, acceptance breakdown
```

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

## Models

| Role | Model | VRAM (FP16) |
|------|-------|------------|
| Target | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | ~4 GB |
| Draft | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | ~2 GB |

Same model family as Baseten's published benchmarks. Both fit on a free T4 (16 GB).
