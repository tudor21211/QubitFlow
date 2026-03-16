"""
Visualisation utilities for benchmarking results and convergence.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "results"


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ───────────────────────────────────────────────────────────────────
#  1. GA convergence plot
# ───────────────────────────────────────────────────────────────────

def plot_ga_convergence(
    history: List[dict],
    title: str = "GA Convergence",
    save: bool = True,
    filename: str = "ga_convergence.png",
):
    """Plot best / average cost per generation."""
    _ensure_dir()
    gens = [h["generation"] for h in history]
    best = [h["best_cost"] for h in history]
    avg = [-h["avg_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, best, label="Best cost", linewidth=2)
    ax.plot(gens, avg, label="Average cost", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  2. Method comparison bar chart
# ───────────────────────────────────────────────────────────────────

def plot_cost_comparison(
    results: List[Dict],
    save: bool = True,
    filename: str = "cost_comparison.png",
):
    """Grouped bar chart: cost for each method on each circuit."""
    _ensure_dir()
    names = [r["name"] for r in results]
    methods = ["original", "baseline", "ga"]
    if any(r.get("hybrid") for r in results):
        methods.append("hybrid")

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))

    for i, m in enumerate(methods):
        vals = []
        for r in results:
            if m in r and r[m] is not None:
                vals.append(r[m]["cost"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=m.capitalize())

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Cost")
    ax.set_title("Cost Comparison Across Methods")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  3. Depth comparison
# ───────────────────────────────────────────────────────────────────

def plot_depth_comparison(
    results: List[Dict],
    save: bool = True,
    filename: str = "depth_comparison.png",
):
    """Bar chart of circuit depth per method."""
    _ensure_dir()
    names = [r["name"] for r in results]
    methods = ["original", "baseline", "ga"]
    if any(r.get("hybrid") for r in results):
        methods.append("hybrid")

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))

    for i, m in enumerate(methods):
        vals = []
        for r in results:
            if m in r and r[m] is not None:
                vals.append(r[m]["depth"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=m.capitalize())

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Depth")
    ax.set_title("Circuit Depth Comparison")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  4. CX gate count comparison
# ───────────────────────────────────────────────────────────────────

def plot_cx_comparison(
    results: List[Dict],
    save: bool = True,
    filename: str = "cx_comparison.png",
):
    """Bar chart of CX gate count per method."""
    _ensure_dir()
    names = [r["name"] for r in results]
    methods = ["original", "baseline", "ga"]
    if any(r.get("hybrid") for r in results):
        methods.append("hybrid")

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))

    for i, m in enumerate(methods):
        vals = []
        for r in results:
            if m in r and r[m] is not None:
                vals.append(r[m]["cx_count"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=m.capitalize())

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("CX Gate Count")
    ax.set_title("CX Gate Count Comparison")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  5. Variance / stability box-plot
# ───────────────────────────────────────────────────────────────────

def plot_stability(
    results: List[Dict],
    save: bool = True,
    filename: str = "stability.png",
):
    """Box-like comparison of GA and Hybrid avg ± std cost."""
    _ensure_dir()
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    width = 0.3

    ga_avg = [r["ga"].get("avg_cost", r["ga"]["cost"]) for r in results]
    ga_std = [r["ga"].get("std_cost", 0) for r in results]
    ax.bar(x - width / 2, ga_avg, width, yerr=ga_std, label="GA", capsize=4)

    if any(r.get("hybrid") for r in results):
        h_avg = [r["hybrid"].get("avg_cost", r["hybrid"]["cost"])
                 if r.get("hybrid") else 0 for r in results]
        h_std = [r["hybrid"].get("std_cost", 0)
                 if r.get("hybrid") else 0 for r in results]
        ax.bar(x + width / 2, h_avg, width, yerr=h_std, label="Hybrid", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Cost (avg ± std)")
    ax.set_title("Optimisation Stability")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  6. Improvement heatmap
# ───────────────────────────────────────────────────────────────────

def plot_improvement_heatmap(
    results: List[Dict],
    save: bool = True,
    filename: str = "improvement_heatmap.png",
):
    """Heatmap of percentage improvement over the original for each
    method × circuit."""
    _ensure_dir()
    names = [r["name"] for r in results]
    methods = ["baseline", "ga"]
    if any(r.get("hybrid") for r in results):
        methods.append("hybrid")

    data = np.zeros((len(methods), len(names)))
    for j, r in enumerate(results):
        orig = r["original"]["cost"]
        for i, m in enumerate(methods):
            if m in r and r[m] is not None and orig > 0:
                data[i, j] = 100.0 * (orig - r[m]["cost"]) / orig

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.capitalize() for m in methods])

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(names)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="black")

    ax.set_title("Improvement over Original (%)")
    fig.colorbar(im, ax=ax, label="% improvement")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
#  Convenience: plot everything
# ───────────────────────────────────────────────────────────────────

def plot_all(results: List[Dict], ga_history: Optional[List[dict]] = None):
    """Generate all plots from benchmark results."""
    if ga_history:
        plot_ga_convergence(ga_history)
    plot_cost_comparison(results)
    plot_depth_comparison(results)
    plot_cx_comparison(results)
    plot_stability(results)
    plot_improvement_heatmap(results)
    print(f"\nAll plots saved to '{OUTPUT_DIR}/' directory.")
