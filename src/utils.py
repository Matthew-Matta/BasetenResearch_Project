"""
Metrics tracking and visualization utilities for speculative decoding benchmarks.
Matches Baseten's metrics vocabulary: TPS, TTFT, E2E latency, acceptance rate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GenerationMetrics:
    tokens_generated: int
    wall_time_s: float
    ttft_s: float                  # time to first token
    tokens_per_second: float
    acceptance_rate: float         # overall (accepted / proposed)
    sa_acceptance_rate: float      # SA-sourced drafts only
    draft_acceptance_rate: float   # draft-model-sourced drafts only
    avg_draft_length: float
    draft_length_history: list[int] = field(default_factory=list)
    source_history: list[str] = field(default_factory=list)  # "SA"|"draft"|"autoregressive"

    def to_dict(self) -> dict:
        return {
            "tokens_generated": self.tokens_generated,
            "wall_time_s": self.wall_time_s,
            "ttft_s": self.ttft_s,
            "tokens_per_second": self.tokens_per_second,
            "acceptance_rate": self.acceptance_rate,
            "sa_acceptance_rate": self.sa_acceptance_rate,
            "draft_acceptance_rate": self.draft_acceptance_rate,
            "avg_draft_length": self.avg_draft_length,
            "draft_length_history": self.draft_length_history,
            "source_history": self.source_history,
        }


class MetricsTracker:
    """Accumulates per-step stats during a generation run."""

    def __init__(self):
        self._ttft: float | None = None
        self._start: float = time.perf_counter()

        # Counters per source
        self._proposed: dict[str, int] = {"SA": 0, "draft": 0, "autoregressive": 0}
        self._accepted: dict[str, int] = {"SA": 0, "draft": 0, "autoregressive": 0}
        self._draft_lengths: list[int] = []
        self._sources: list[str] = []

    def record_ttft(self, elapsed: float) -> None:
        if self._ttft is None:
            self._ttft = elapsed

    def record_draft_attempt(
        self,
        source: Literal["SA", "draft", "autoregressive"],
        proposed: int,
        accepted: int,
        draft_len: int,
    ) -> None:
        self._proposed[source] += proposed
        self._accepted[source] += accepted
        self._draft_lengths.append(draft_len)
        self._sources.append(source)

    def finalize(self, total_tokens: int, total_time: float) -> GenerationMetrics:
        total_proposed = sum(self._proposed.values())
        total_accepted = sum(self._accepted.values())

        def _rate(src: str) -> float:
            p = self._proposed[src]
            return self._accepted[src] / p if p > 0 else 0.0

        return GenerationMetrics(
            tokens_generated=total_tokens,
            wall_time_s=total_time,
            ttft_s=self._ttft or 0.0,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0.0,
            acceptance_rate=total_accepted / total_proposed if total_proposed > 0 else 0.0,
            sa_acceptance_rate=_rate("SA"),
            draft_acceptance_rate=_rate("draft"),
            avg_draft_length=float(np.mean(self._draft_lengths)) if self._draft_lengths else 0.0,
            draft_length_history=list(self._draft_lengths),
            source_history=list(self._sources),
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark_results(results: dict[str, list[GenerationMetrics]], output_dir: str) -> None:
    """
    results: method_name → list of GenerationMetrics (one per prompt)

    Generates three figures:
      1. tps_comparison.png — bar chart of mean TPS per method
      2. dynamic_draft_length.png — draft length over steps for hybrid_dynamic
      3. acceptance_rate_breakdown.png — SA vs draft acceptance rates per method
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # 1. TPS bar chart
    # ------------------------------------------------------------------
    mean_tps = [np.mean([m.tokens_per_second for m in results[method]]) for method in methods]
    std_tps = [np.std([m.tokens_per_second for m in results[method]]) for method in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, mean_tps, yerr=std_tps, capsize=4, color=colors)
    ax.set_ylabel("Tokens per Second (TPS)")
    ax.set_title("Throughput Comparison Across Decoding Methods")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    for bar, v in zip(bars, mean_tps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tps_comparison.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Dynamic draft length over steps (hybrid_dynamic only)
    # ------------------------------------------------------------------
    if "hybrid_dynamic" in results and results["hybrid_dynamic"]:
        # Use first sample's history
        history = results["hybrid_dynamic"][0].draft_length_history
        if history:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history, marker="o", markersize=3, linewidth=1, label="draft length")
            ax.axhline(np.mean(history), color="red", linestyle="--", label=f"mean={np.mean(history):.2f}")
            ax.set_xlabel("Decoding Step")
            ax.set_ylabel("Draft Length")
            ax.set_title("Dynamic Draft Length Over Steps (hybrid_dynamic)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "dynamic_draft_length.png"), dpi=150)
            plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Acceptance rate breakdown
    # ------------------------------------------------------------------
    sa_rates = [np.mean([m.sa_acceptance_rate for m in results[method]]) for method in methods]
    draft_rates = [np.mean([m.draft_acceptance_rate for m in results[method]]) for method in methods]
    overall_rates = [np.mean([m.acceptance_rate for m in results[method]]) for method in methods]

    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, sa_rates, width, label="SA acceptance rate", color="steelblue")
    ax.bar(x, draft_rates, width, label="Draft model acceptance rate", color="darkorange")
    ax.bar(x + width, overall_rates, width, label="Overall acceptance rate", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rate Breakdown by Source")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "acceptance_rate_breakdown.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {output_dir}/")


def print_summary_table(results: dict[str, list[GenerationMetrics]]) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Method':<22} {'TPS':>8} {'TTFT(s)':>9} {'Accept%':>9} {'SA%':>7} {'Draft%':>8} {'AvgDraftLen':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for method, metrics_list in results.items():
        tps = np.mean([m.tokens_per_second for m in metrics_list])
        ttft = np.mean([m.ttft_s for m in metrics_list])
        acc = np.mean([m.acceptance_rate for m in metrics_list]) * 100
        sa = np.mean([m.sa_acceptance_rate for m in metrics_list]) * 100
        draft = np.mean([m.draft_acceptance_rate for m in metrics_list]) * 100
        adl = np.mean([m.avg_draft_length for m in metrics_list])
        print(f"{method:<22} {tps:>8.2f} {ttft:>9.4f} {acc:>8.1f}% {sa:>6.1f}% {draft:>7.1f}% {adl:>12.2f}")
    print("=" * len(header) + "\n")
