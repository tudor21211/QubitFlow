"""Plot and save before/after artifacts for interactive visualization."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.quantum_info import Operator, Statevector, random_statevector, state_fidelity

OUTPUT_DIR = "results"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_run_prefix() -> str:
    """Build a unique filename prefix for each app run."""
    return datetime.now().strftime("viz_%Y%m%d_%H%M%S")


def save_circuit_image(circuit, file_path: str, title: str) -> str:
    """Save a Matplotlib circuit diagram to disk."""
    try:
        fig = circuit.draw(output="mpl", fold=-1)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        return file_path
    except MissingOptionalLibraryError:
        # Fallback: render ASCII diagram into an image when mpl drawer deps are missing.
        ascii_diagram = str(circuit.draw(output="text"))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis("off")
        ax.set_title(f"{title} (ASCII fallback)")
        ax.text(0.01, 0.99, ascii_diagram, va="top", ha="left", family="monospace")
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
    return file_path


def save_metrics_comparison(
    before_metrics: Dict,
    after_metrics: Dict,
    file_path: str,
    method_used: str,
) -> str:
    """Save a compact bar chart for key before/after metrics."""
    labels = ["Total gates", "Depth", "CX gates", "Cost"]
    before_values = [
        before_metrics["total_gates"],
        before_metrics["depth"],
        before_metrics["cx_count"],
        before_metrics["cost"],
    ]
    after_values = [
        after_metrics["total_gates"],
        after_metrics["depth"],
        after_metrics["cx_count"],
        after_metrics["cost"],
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], before_values, width, label="Before")
    ax.bar([i + width / 2 for i in x], after_values, width, label=f"After ({method_used})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Circuit Optimization Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path


def save_equivalence_plot(
    original_circuit,
    optimized_circuit,
    equivalence_result: Dict,
    file_path: str,
) -> str:
    """
    Save a visual equivalence comparison between two circuits.

    For circuits with ≤ 8 qubits (unitary method) a side-by-side heatmap of
    |U_original|, |U_optimized| and the element-wise difference is shown.
    For larger circuits (statevector method) a state-fidelity bar chart is
    shown instead.
    """
    is_equiv = equivalence_result["is_equivalent"]
    method = equivalence_result["method"]
    status_str = "EQUIVALENT" if is_equiv else "NOT EQUIVALENT"
    status_color = "green" if is_equiv else "red"

    def _clean(qc):
        c = qc.copy()
        c.remove_final_measurements(inplace=True)
        return c

    if method == "unitary":
        c1 = _clean(original_circuit)
        c2 = _clean(optimized_circuit)
        mat1 = Operator(c1).data
        mat2 = Operator(c2).data

        # Remove global phase before computing the difference
        overlap = np.trace(mat1.conj().T @ mat2)
        if abs(overlap) > 1e-12:
            phase = overlap / abs(overlap)
            mat2_aligned = mat2 * np.conj(phase)
        else:
            mat2_aligned = mat2

        abs1 = np.abs(mat1)
        abs2 = np.abs(mat2_aligned)
        diff = np.abs(mat1 - mat2_aligned)
        max_diff = equivalence_result["max_abs_diff"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            f"Unitary Equivalence Check — {status_str}  (max element diff = {max_diff:.2e})",
            fontsize=13,
            color=status_color,
            fontweight="bold",
        )

        im0 = axes[0].imshow(abs1, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[0].set_title("Original  |U|", fontsize=11)
        axes[0].set_xlabel("Column index")
        axes[0].set_ylabel("Row index")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(abs2, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[1].set_title("Optimized  |U|  (phase-aligned)", fontsize=11)
        axes[1].set_xlabel("Column index")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        vmax_diff = max(diff.max(), 1e-10)
        im2 = axes[2].imshow(diff, vmin=0, vmax=vmax_diff, cmap="Reds", aspect="auto")
        axes[2].set_title(f"Difference  |U₁ − U₂|", fontsize=11)
        axes[2].set_xlabel("Column index")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

    else:  # statevector
        fidelities = equivalence_result.get("fidelities", [])
        if not fidelities:
            # Fallback: re-run sampling so the plot always has data
            rng = np.random.default_rng(42)
            c1 = _clean(original_circuit)
            c2 = _clean(optimized_circuit)
            n = c1.num_qubits
            fidelities = []
            psi0 = Statevector.from_int(0, 2**n)
            fidelities.append(float(state_fidelity(psi0.evolve(c1), psi0.evolve(c2), validate=False)))
            for _ in range(7):
                psi = random_statevector(2**n, seed=int(rng.integers(0, 2**31)))
                fidelities.append(float(state_fidelity(psi.evolve(c1), psi.evolve(c2), validate=False)))

        labels = ["|0⋯0⟩"] + [f"rand {i}" for i in range(1, len(fidelities))]
        colors = ["#2ca02c" if f > 1 - 1e-6 else "#d62728" for f in fidelities]
        min_f = min(fidelities)
        max_err = equivalence_result.get("max_fidelity_error", 1 - min_f)

        fig, ax = plt.subplots(figsize=(max(7, len(fidelities) * 1.1), 4.5))
        bars = ax.bar(labels, fidelities, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=0.8, label="Perfect fidelity = 1")
        ax.axhline(
            y=1 - 1e-6, color="#ff7f0e", linestyle=":", linewidth=0.8, label="Tolerance threshold"
        )

        # Annotate each bar with its value
        for bar, f in zip(bars, fidelities):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.002,
                f"{f:.6f}",
                ha="center",
                va="top",
                fontsize=8,
                color="white" if f > 0.5 else "black",
            )

        ax.set_ylim([max(0, min_f - 0.02), 1.008])
        ax.set_ylabel("State fidelity  |\u27e8ψ₁|ψ₂⟩|²", fontsize=11)
        ax.set_xlabel("Input state", fontsize=11)
        ax.set_title(
            f"Statevector Fidelity Check \u2014 {status_str}  (max error = {max_err:.2e})",
            fontsize=13,
            color=status_color,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()

    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return file_path


def save_artifacts(result) -> Dict[str, str]:
    """Persist before/after diagrams and metrics chart."""
    _ensure_output_dir()
    prefix = make_run_prefix()

    before_path = os.path.join(OUTPUT_DIR, f"{prefix}_before.png")
    after_path = os.path.join(OUTPUT_DIR, f"{prefix}_after.png")
    metrics_path = os.path.join(OUTPUT_DIR, f"{prefix}_metrics.png")
    equiv_path = os.path.join(OUTPUT_DIR, f"{prefix}_equivalence.png")
    ga_only_path = None
    hybrid_path = None

    save_circuit_image(result.original_circuit, before_path, "Original circuit")
    save_circuit_image(result.optimized_circuit, after_path, f"Optimized circuit ({result.method_used})")
    save_metrics_comparison(
        before_metrics=result.original_metrics,
        after_metrics=result.optimized_metrics,
        file_path=metrics_path,
        method_used=result.method_used,
    )
    if result.equivalence_result is not None:
        save_equivalence_plot(
            original_circuit=result.original_circuit,
            optimized_circuit=result.optimized_circuit,
            equivalence_result=result.equivalence_result,
            file_path=equiv_path,
        )
    else:
        equiv_path = None

    ga_metrics = getattr(result, "ga_metrics", None)
    hybrid_metrics = getattr(result, "hybrid_metrics", None)
    ga_circuit = getattr(result, "ga_circuit", None)
    hybrid_circuit = getattr(result, "hybrid_circuit", None)

    if (
        ga_metrics is not None
        and hybrid_metrics is not None
        and ga_circuit is not None
        and hybrid_circuit is not None
        and hybrid_metrics["cost"] < ga_metrics["cost"]
    ):
        ga_only_path = os.path.join(OUTPUT_DIR, f"{prefix}_ga_only.png")
        hybrid_path = os.path.join(OUTPUT_DIR, f"{prefix}_hybrid.png")
        save_circuit_image(ga_circuit, ga_only_path, "GA-only optimized circuit")
        save_circuit_image(hybrid_circuit, hybrid_path, "Hybrid (GA+RL) optimized circuit")

    return {
        "before": before_path,
        "after": after_path,
        "metrics": metrics_path,
        "equivalence": equiv_path,
        "ga_only": ga_only_path,
        "hybrid": hybrid_path,
    }
