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
#  7. Pareto front scatter plots (thesis-oriented)
# ───────────────────────────────────────────────────────────────────

def plot_pareto_front(
    results: List[Dict],
    objective_x: str = "depth",
    objective_y: str = "cx_count",
    methods: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "pareto_front.png",
):
    """Scatter objective trade-offs for selected methods.

    Each point is one circuit result for one method.
    """
    _ensure_dir()
    methods = methods or ["ga", "hybrid"]
    color_map = {
        "original": "#7f8c8d",
        "baseline": "#2980b9",
        "ga": "#d35400",
        "hybrid": "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = False

    for method in methods:
        xs = []
        ys = []
        labels = []
        for r in results:
            method_res = r.get(method)
            if not method_res:
                continue
            obj = method_res.get("objectives", {})
            if objective_x in obj and objective_y in obj:
                xs.append(obj[objective_x])
                ys.append(obj[objective_y])
                labels.append(r.get("name", ""))

        if xs:
            plotted_any = True
            ax.scatter(
                xs,
                ys,
                s=80,
                alpha=0.8,
                label=method.capitalize(),
                color=color_map.get(method, None),
            )
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=8, alpha=0.85)

    ax.set_xlabel(objective_x.replace("_", " ").title())
    ax.set_ylabel(objective_y.replace("_", " ").title())
    ax.set_title(f"Pareto View: {objective_y} vs {objective_x}")
    if plotted_any:
        ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    plt.close(fig)


def plot_pareto_ablation(
    results: List[Dict],
    objective_x: str = "depth",
    objective_y: str = "cx_count",
    save: bool = True,
    filename: str = "pareto_ablation.png",
):
    """Plot weighted vs pareto ablation points for GA/Hybrid.

    Requires benchmark rows containing results["ablation"].
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(9, 6))

    style_map = {
        "ga-weighted": ("o", "#e67e22"),
        "ga-pareto": ("^", "#d35400"),
        "hybrid-weighted": ("s", "#2ecc71"),
        "hybrid-pareto": ("D", "#27ae60"),
    }

    series = {k: {"x": [], "y": []} for k in style_map}
    for r in results:
        for row in r.get("ablation", []):
            mode = row.get("selection_mode", "").lower()
            for method in ("ga", "hybrid"):
                method_res = row.get(method)
                if not method_res:
                    continue
                obj = method_res.get("objectives", {})
                if objective_x not in obj or objective_y not in obj:
                    continue
                key = f"{method}-{mode}"
                if key in series:
                    series[key]["x"].append(obj[objective_x])
                    series[key]["y"].append(obj[objective_y])

    plotted_any = False
    for key, values in series.items():
        if not values["x"]:
            continue
        marker, color = style_map[key]
        ax.scatter(values["x"], values["y"], marker=marker, color=color,
                   s=90, alpha=0.82, label=key.replace("-", " ").title())
        plotted_any = True

    ax.set_xlabel(objective_x.replace("_", " ").title())
    ax.set_ylabel(objective_y.replace("_", " ").title())
    ax.set_title("Selection Mode Ablation Pareto View")
    if plotted_any:
        ax.legend()
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
    plot_pareto_front(results, objective_x="depth", objective_y="cx_count",
                      filename="pareto_depth_vs_cx.png")
    plot_pareto_front(results, objective_x="execution_time_norm",
                      objective_y="estimated_error",
                      filename="pareto_time_vs_error.png")
    if any(r.get("ablation") for r in results):
        plot_pareto_ablation(results, objective_x="depth",
                             objective_y="cx_count",
                             filename="pareto_ablation_depth_vs_cx.png")
    print(f"\nAll plots saved to '{OUTPUT_DIR}/' directory.")
