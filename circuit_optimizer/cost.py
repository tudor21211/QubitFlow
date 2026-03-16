"""
Cost / fitness function for quantum circuits.

    C = alpha * depth  +  beta * CX_count  +  gamma * estimated_error

The GA uses  fitness = -C   (minimise cost).
The RL uses  reward  = C_before - C_after  (improvement).
"""

from __future__ import annotations

from collections.abc import Iterable
from qiskit import QuantumCircuit

import config


def circuit_cost(qc: QuantumCircuit,
                 alpha: float | None = None,
                 beta: float | None = None,
                 gamma: float | None = None,
                 delta: float | None = None,
                 epsilon: float | None = None,
                 zeta: float | None = None,
                 hardware_profile: str | None = None,
                 objective_preset: str | None = None) -> float:
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
    weights = _resolve_objective_weights(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon,
        zeta=zeta,
        objective_preset=objective_preset,
    )
    obj = circuit_objectives(qc, hardware_profile=hardware_profile)

    return (
        weights["alpha_depth"] * obj["depth"]
        + weights["beta_cnot"] * obj["cx_count"]
        + weights["gamma_error"] * obj["estimated_error"]
        + weights["delta_exec_time"] * obj["execution_time_norm"]
        + weights["epsilon_routing"] * obj["routing_penalty"]
        + weights["zeta_coherence"] * obj["coherence_penalty"]
    )


def circuit_metrics(qc: QuantumCircuit) -> dict:
    """Return a dict of individual metrics for reporting."""
    obj = circuit_objectives(qc)
    return {
        "depth": obj["depth"],
        "total_gates": obj["total_gates"],
        "cx_count": obj["cx_count"],
        "estimated_error": obj["estimated_error"],
        "execution_time_ns": obj["execution_time_ns"],
        "execution_time_norm": obj["execution_time_norm"],
        "routing_penalty": obj["routing_penalty"],
        "coherence_penalty": obj["coherence_penalty"],
        "cost": circuit_cost(qc),
    }


def circuit_objectives(
    qc: QuantumCircuit,
    hardware_profile: str | None = None,
) -> dict:
    """Return decomposed objective components for multi-objective analysis."""
    profile = _hardware_profile(hardware_profile)
    ops = qc.count_ops()

    depth = qc.depth()
    total_gates = sum(ops.values())
    cx_count = ops.get("cx", 0)
    estimated_error = _estimated_error(qc, profile=profile)
    execution_time_ns = _estimate_execution_time_ns(qc, profile=profile)
    execution_time_norm = execution_time_ns / 1_000.0  # us-scale proxy
    routing_penalty = _routing_pressure_penalty(qc, profile=profile)
    coherence_penalty = _coherence_penalty(execution_time_ns, profile=profile)

    return {
        "depth": depth,
        "total_gates": total_gates,
        "cx_count": cx_count,
        "estimated_error": estimated_error,
        "execution_time_ns": execution_time_ns,
        "execution_time_norm": execution_time_norm,
        "routing_penalty": routing_penalty,
        "coherence_penalty": coherence_penalty,
    }


# ── Simple error model ────────────────────────────────────────────

def _estimated_error(qc: QuantumCircuit, profile: dict | None = None) -> float:
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

    p = profile or _hardware_profile(None)
    rates = p.get("gate_error_rates", {})
    single_err = rates.get("single", config.SINGLE_GATE_ERROR)
    two_err = rates.get("two", config.TWO_GATE_ERROR)
    three_err = rates.get("three", 3 * config.TWO_GATE_ERROR)

    err = (single_gates * single_err
           + two_gates * two_err
           + three_gates * three_err
           + qc.num_qubits * config.READOUT_ERROR)
    return err


def _estimate_execution_time_ns(qc: QuantumCircuit, profile: dict) -> float:
    """Approximate execution time proxy from gate arity classes."""
    ops = qc.count_ops()
    durations = profile.get("gate_durations_ns", {})

    single_ns = float(durations.get("single", 35.0))
    two_ns = float(durations.get("two", 250.0))
    three_ns = float(durations.get("three", 700.0))
    measure_ns = float(durations.get("measure", 700.0))

    single_gates = sum(v for k, v in ops.items()
                       if k not in ("cx", "cz", "swap", "ecr",
                                    "rzz", "rxx", "ryy",
                                    "ccx", "cswap", "barrier", "measure"))
    two_gates = sum(ops.get(g, 0) for g in ("cx", "cz", "swap", "ecr",
                                            "rzz", "rxx", "ryy"))
    three_gates = sum(ops.get(g, 0) for g in ("ccx", "cswap"))
    measures = ops.get("measure", 0)

    return (
        single_gates * single_ns
        + two_gates * two_ns
        + three_gates * three_ns
        + measures * measure_ns
    )


def _routing_pressure_penalty(qc: QuantumCircuit, profile: dict) -> float:
    """
    Soft penalty for 2-qubit interactions not present in coupling map.
    0 means no pressure or no map configured.
    """
    coupling_map = profile.get("coupling_map", [])
    if not coupling_map:
        return 0.0

    edges = _normalize_edge_set(coupling_map)
    if not edges:
        return 0.0

    total_two = 0
    invalid_two = 0
    for inst, qargs, _ in qc.data:
        if len(qargs) != 2:
            continue
        total_two += 1
        q0 = qc.find_bit(qargs[0]).index
        q1 = qc.find_bit(qargs[1]).index
        if tuple(sorted((q0, q1))) not in edges:
            invalid_two += 1

    if total_two == 0:
        return 0.0
    return invalid_two / total_two


def _coherence_penalty(execution_time_ns: float, profile: dict) -> float:
    """Simple coherence penalty proxy from execution time relative to T1/T2."""
    t1 = float(profile.get("avg_t1_ns", 0.0) or 0.0)
    t2 = float(profile.get("avg_t2_ns", 0.0) or 0.0)
    if t1 <= 0.0 or t2 <= 0.0:
        return 0.0
    return execution_time_ns / min(t1, t2)


def _normalize_edge_set(coupling_map: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    edges = set()
    for edge in coupling_map:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        if a == b:
            continue
        edges.add(tuple(sorted((a, b))))
    return edges


def _hardware_profile(name: str | None) -> dict:
    profile_name = name or config.ACTIVE_HARDWARE_PROFILE
    profiles = getattr(config, "HARDWARE_PROFILES", {})
    return profiles.get(profile_name, profiles.get("sim_default", {}))


def _resolve_objective_weights(
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
    delta: float | None,
    epsilon: float | None,
    zeta: float | None,
    objective_preset: str | None,
) -> dict:
    presets = getattr(config, "OBJECTIVE_PRESETS", {})
    preset_name = objective_preset or getattr(config, "ACTIVE_OBJECTIVE_PRESET", "legacy")
    preset = presets.get(preset_name, {})

    return {
        "alpha_depth": alpha if alpha is not None else preset.get("alpha_depth", config.ALPHA_DEPTH),
        "beta_cnot": beta if beta is not None else preset.get("beta_cnot", config.BETA_CNOT),
        "gamma_error": gamma if gamma is not None else preset.get("gamma_error", config.GAMMA_ERROR),
        "delta_exec_time": delta if delta is not None else preset.get("delta_exec_time", getattr(config, "DELTA_EXEC_TIME", 0.0)),
        "epsilon_routing": epsilon if epsilon is not None else preset.get("epsilon_routing", getattr(config, "EPSILON_ROUTING", 0.0)),
        "zeta_coherence": zeta if zeta is not None else preset.get("zeta_coherence", getattr(config, "ZETA_COHERENCE", 0.0)),
    }


def fitness(qc: QuantumCircuit) -> float:
    """GA fitness (higher is better) = -cost."""
    return -circuit_cost(qc)


def reward(cost_before: float, cost_after: float) -> float:
    """RL reward for a single step."""
    return cost_before - cost_after
