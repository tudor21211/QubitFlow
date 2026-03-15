"""
Baseline transpiler optimisation using Qiskit's built-in passes.

Provides a reference point for comparison with the GA / RL / Hybrid
approaches.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.representation import copy_circuit


def transpile_baseline(
    qc: QuantumCircuit,
    optimization_level: int = 3,
    seed: int = 42,
) -> QuantumCircuit:
    """
    Optimise *qc* with the Qiskit transpiler at the given level.

    Parameters
    ----------
    qc : QuantumCircuit
    optimization_level : 0-3  (3 = heaviest optimisation)
    seed : random seed for reproducibility

    Returns
    -------
    Optimised QuantumCircuit
    """
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    optimized = pm.run(qc)
    return optimized


def baseline_metrics(qc: QuantumCircuit, optimization_level: int = 3) -> dict:
    """Return cost + metrics after standard transpilation."""
    opt = transpile_baseline(qc, optimization_level=optimization_level)
    m = circuit_metrics(opt)
    m["method"] = f"qiskit_O{optimization_level}"
    return m, opt
