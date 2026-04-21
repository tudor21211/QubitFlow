"""
Toffoli decomposition utilities.

Implements multiple CCX decomposition variants and a circuit-wide
replacement helper that chooses the best variant according to a
selection objective.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from qiskit import QuantumCircuit


ToffoliMeta = Dict[str, object]


def _meta(circuit: QuantumCircuit, variant: str) -> ToffoliMeta:
    return {
        "circuit": circuit,
        "depth": int(circuit.depth() or 0),
        "gate_count": int(len(circuit.data)),
        "variant": variant,
    }


def decompose_toffoli_standard(
    circuit: QuantumCircuit,
    q_controls: List[int],
    q_target: int,
) -> ToffoliMeta:
    """Append a no-ancilla textbook-style Toffoli decomposition."""
    c0, c1 = q_controls
    circuit.h(q_target)
    circuit.cx(c1, q_target)
    circuit.tdg(q_target)
    circuit.cx(c0, q_target)
    circuit.t(q_target)
    circuit.cx(c1, q_target)
    circuit.tdg(q_target)
    circuit.cx(c0, q_target)
    circuit.t(c1)
    circuit.t(q_target)
    circuit.h(q_target)
    circuit.cx(c0, c1)
    circuit.t(c0)
    circuit.tdg(c1)
    circuit.cx(c0, c1)
    return _meta(circuit, "standard")


def decompose_toffoli_ancilla(
    circuit: QuantumCircuit,
    q_controls: List[int],
    q_target: int,
    ancilla: int,
) -> ToffoliMeta:
    """Append an exact ancilla-assisted decomposition.

    This computes the control conjunction into ancilla, applies it to target,
    then uncomputes ancilla.
    """
    c0, c1 = q_controls
    circuit.ccx(c0, c1, ancilla)
    circuit.cx(ancilla, q_target)
    circuit.ccx(c0, c1, ancilla)
    return _meta(circuit, "ancilla")


def decompose_toffoli_relative_phase(
    circuit: QuantumCircuit,
    q_controls: List[int],
    q_target: int,
) -> ToffoliMeta:
    """Append a relative-phase Toffoli (RCCX)."""
    c0, c1 = q_controls
    circuit.rccx(c0, c1, q_target)
    return _meta(circuit, "relative_phase")


def _choose_candidate(
    candidates: List[ToffoliMeta],
    objective: str,
    fitness_fn: Optional[Callable[[ToffoliMeta], float]] = None,
) -> ToffoliMeta:
    if objective == "gate_count":
        return min(candidates, key=lambda c: (c["gate_count"], c["depth"]))
    if objective == "fitness" and fitness_fn is not None:
        return max(candidates, key=fitness_fn)
    return min(candidates, key=lambda c: (c["depth"], c["gate_count"]))


def replace_toffoli_with_best(
    qc: QuantumCircuit,
    objective: str = "depth",
    allow_relative_phase: bool = False,
    allow_ancilla_auto: bool = False,
    fitness_fn: Optional[Callable[[ToffoliMeta], float]] = None,
) -> tuple[QuantumCircuit, List[dict]]:
    """
    Replace every CCX gate with the best decomposition variant.

    Returns
    -------
    (new_circuit, replacement_log)
    """
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    replacement_log: List[dict] = []

    for inst, qargs, cargs in qc.data:
        if inst.name != "ccx":
            new_qc.append(inst, qargs, cargs)
            continue

        controls = [qargs[0], qargs[1]]
        target = qargs[2]
        available_ancilla = [q for q in qc.qubits if q not in controls and q != target]

        standard_local = QuantumCircuit(3)
        standard_meta = decompose_toffoli_standard(standard_local, [0, 1], 2)
        candidates: List[ToffoliMeta] = [standard_meta]

        anc_meta: Optional[ToffoliMeta] = None
        if allow_ancilla_auto and available_ancilla:
            anc_local = QuantumCircuit(4)
            anc_meta = decompose_toffoli_ancilla(anc_local, [0, 1], 2, 3)
            candidates.append(anc_meta)

        rel_meta: Optional[ToffoliMeta] = None
        if allow_relative_phase:
            rel_local = QuantumCircuit(3)
            rel_meta = decompose_toffoli_relative_phase(rel_local, [0, 1], 2)
            candidates.append(rel_meta)

        selected = _choose_candidate(candidates, objective=objective, fitness_fn=fitness_fn)

        if selected is standard_meta:
            new_qc.append(standard_meta["circuit"].to_instruction(), controls + [target])
            chosen = "standard"
        elif anc_meta is not None and selected is anc_meta:
            anc = available_ancilla[0]
            new_qc.append(anc_meta["circuit"].to_instruction(), controls + [target, anc])
            chosen = "ancilla"
        else:
            new_qc.append(rel_meta["circuit"].to_instruction(), controls + [target])
            chosen = "relative_phase"

        replacement_log.append(
            {
                "variant": chosen,
                "depth": int(selected["depth"]),
                "gate_count": int(selected["gate_count"]),
            }
        )

    return new_qc, replacement_log