"""
Cost / fitness function for quantum circuits.

    C = alpha * depth  +  beta * CX_count  +  gamma * estimated_error

The GA uses  fitness = -C   (minimise cost).
The RL uses  reward  = C_before - C_after  (improvement).
"""

from __future__ import annotations

from qiskit import QuantumCircuit

import config
from circuit_optimizer.toffoli import replace_toffoli_with_best


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
    # Keep signature compatibility, but the active objective is the
    # updated configurable fitness formula from the requirements.
    return -fitness(qc)


def circuit_metrics(qc: QuantumCircuit) -> dict:
    """Return a dict of individual metrics for reporting."""
    toffoli_normalized, _ = replace_toffoli_with_best(
        qc,
        objective=config.TOFFOLI_SELECTION_OBJECTIVE,
        allow_relative_phase=config.TOFFOLI_ALLOW_RELATIVE_PHASE,
        allow_ancilla_auto=config.TOFFOLI_ALLOW_ANCILLA_AUTO,
    )
    ops = toffoli_normalized.count_ops()
    depth = compute_depth(toffoli_normalized)
    delay = compute_total_delay(toffoli_normalized)
    fidelity = circuit_fidelity(toffoli_normalized)

    return {
        "depth": depth,
        "total_gates": sum(ops.values()),
        "cx_count": ops.get("cx", 0),
        "delay": delay,
        "fidelity": fidelity,
        "estimated_error": _estimated_error(toffoli_normalized),
        "fitness": fitness(toffoli_normalized),
        "cost": circuit_cost(qc),
    }


def compute_depth(qc: QuantumCircuit) -> int:
    """
    Parallel-aware depth estimate by tracking latest occupied layer per qubit.
    """
    if qc.num_qubits == 0:
        return 0
    last_layer = [0 for _ in range(qc.num_qubits)]
    max_layer = 0

    for inst, qargs, _ in qc.data:
        if len(qargs) == 0 or inst.name in {"barrier", "measure"}:
            continue
        qubits = [qc.find_bit(q).index for q in qargs]
        layer = max(last_layer[q] for q in qubits) + 1
        for q in qubits:
            last_layer[q] = layer
        if layer > max_layer:
            max_layer = layer
    return max_layer


def _gate_delay(inst_name: str, arity: int) -> int:
    if inst_name in {"cx", "cz", "cy", "ecr"}:
        return config.DELAY_CNOT
    if inst_name in {"ccx", "cswap"} or arity >= 3:
        return config.DELAY_TOFFOLI
    return config.DELAY_SINGLE_QUBIT


def compute_total_delay(qc: QuantumCircuit) -> int:
    total = 0
    for inst, qargs, _ in qc.data:
        if inst.name in {"barrier", "measure"}:
            continue
        total += _gate_delay(inst.name, len(qargs))
    return int(total)


def circuit_fidelity(qc: QuantumCircuit) -> float:
    return max(0.0, 1.0 - _estimated_error(qc))


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


def fitness(
    qc: QuantumCircuit,
    w1: float | None = None,
    w2: float | None = None,
    w3: float | None = None,
    w4: float | None = None,
) -> float:
    """
    Updated configurable fitness:
        + w1 * fidelity - w2 * depth - w3 * delay - w4 * gate_count
    """
    if w1 is None:
        w1 = config.FITNESS_W1_FIDELITY
    if w2 is None:
        w2 = config.FITNESS_W2_DEPTH
    if w3 is None:
        w3 = config.FITNESS_W3_DELAY
    if w4 is None:
        w4 = config.FITNESS_W4_GATE_COUNT

    normalized, _ = replace_toffoli_with_best(
        qc,
        objective=config.TOFFOLI_SELECTION_OBJECTIVE,
        allow_relative_phase=config.TOFFOLI_ALLOW_RELATIVE_PHASE,
        allow_ancilla_auto=config.TOFFOLI_ALLOW_ANCILLA_AUTO,
    )
    fidelity = circuit_fidelity(normalized)
    depth = compute_depth(normalized)
    delay = compute_total_delay(normalized)
    gate_count = len(normalized.data)
    return (w1 * fidelity) - (w2 * depth) - (w3 * delay) - (w4 * gate_count)


def reward(cost_before: float, cost_after: float) -> float:
    """RL reward for a single step."""
    return cost_before - cost_after
