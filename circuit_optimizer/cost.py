"""
Cost / fitness function for quantum circuits.

    C = alpha * depth  +  beta * CX_count  +  gamma * estimated_error

The GA uses  fitness = -C   (minimise cost).
The RL uses  reward  = C_before - C_after  (improvement).
"""

from __future__ import annotations

from qiskit import QuantumCircuit

import config


def circuit_cost(qc: QuantumCircuit,
                 alpha: float | None = None,
                 beta: float | None = None,
                 gamma: float | None = None) -> float:
    """
    Scalar cost of a circuit.

    Parameters
    ----------
    qc : QuantumCircuit
    alpha, beta, gamma : override weights (default from config)

    Returns
    -------
    float  –  lower is better
    """
    if alpha is None:
        alpha = config.ALPHA_DEPTH
    if beta is None:
        beta = config.BETA_CNOT
    if gamma is None:
        gamma = config.GAMMA_ERROR

    depth = qc.depth()
    ops = qc.count_ops()
    cx_count = ops.get("cx", 0)
    error = _estimated_error(qc)

    return alpha * depth + beta * cx_count + gamma * error


def circuit_metrics(qc: QuantumCircuit) -> dict:
    """Return a dict of individual metrics for reporting."""
    ops = qc.count_ops()
    return {
        "depth": qc.depth(),
        "total_gates": sum(ops.values()),
        "cx_count": ops.get("cx", 0),
        "estimated_error": _estimated_error(qc),
        "cost": circuit_cost(qc),
    }


# ── Simple error model ────────────────────────────────────────────

def _estimated_error(qc: QuantumCircuit) -> float:
    """
    Rough estimate of total depolarising error.

    P(no error) ≈ prod over gates of (1 - e_gate)
    total_error ≈ 1 - P(no error)

    For small errors this is approximately the sum of per-gate errors.
    """
    ops = qc.count_ops()
    single_gates = sum(v for k, v in ops.items()
                       if k not in ("cx", "cz", "swap", "ecr",
                                    "rzz", "rxx", "ryy",
                                    "ccx", "cswap", "barrier", "measure"))
    two_gates = sum(ops.get(g, 0) for g in ("cx", "cz", "swap", "ecr",
                                              "rzz", "rxx", "ryy"))
    three_gates = sum(ops.get(g, 0) for g in ("ccx", "cswap"))

    err = (single_gates * config.SINGLE_GATE_ERROR
           + two_gates * config.TWO_GATE_ERROR
           + three_gates * 3 * config.TWO_GATE_ERROR   # pessimistic
           + qc.num_qubits * config.READOUT_ERROR)
    return err


def fitness(qc: QuantumCircuit) -> float:
    """GA fitness (higher is better) = -cost."""
    return -circuit_cost(qc)


def reward(cost_before: float, cost_after: float) -> float:
    """RL reward for a single step."""
    return cost_before - cost_after
