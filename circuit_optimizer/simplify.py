"""
Circuit simplification and identity-based refinement.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate


def _same_qubits(a_qargs, b_qargs) -> bool:
    if len(a_qargs) != len(b_qargs):
        return False
    return all(qa == qb for qa, qb in zip(a_qargs, b_qargs))


def _angle_of(inst) -> float | None:
    if not inst.params:
        return None
    try:
        return float(inst.params[0])
    except Exception:
        return None


def _is_identity_rotation(name: str, angle: float, eps: float = 1e-10) -> bool:
    if name not in {"rz", "rx", "ry"}:
        return False
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return abs(wrapped) < eps


def _apply_rules_once(qc: QuantumCircuit) -> tuple[QuantumCircuit, bool]:
    data = list(qc.data)
    out = []
    i = 0
    changed = False

    while i < len(data):
        inst, qargs, cargs = data[i]

        if inst.name == "id":
            changed = True
            i += 1
            continue

        if inst.name in {"rx", "ry", "rz"}:
            a = _angle_of(inst)
            if a is not None and _is_identity_rotation(inst.name, a):
                changed = True
                i += 1
                continue

        if i + 1 < len(data):
            nxt_inst, nxt_qargs, nxt_cargs = data[i + 1]

            if _same_qubits(qargs, nxt_qargs):
                # Self-inverse cancellations
                if inst.name in {"x", "h"} and nxt_inst.name == inst.name:
                    changed = True
                    i += 2
                    continue
                if inst.name == "cx" and nxt_inst.name == "cx":
                    changed = True
                    i += 2
                    continue

                # Rotation merging
                if inst.name == nxt_inst.name and inst.name in {"rz", "rx", "ry"}:
                    a1 = _angle_of(inst)
                    a2 = _angle_of(nxt_inst)
                    if a1 is not None and a2 is not None:
                        angle = a1 + a2
                        if inst.name == "rz":
                            out.append((RZGate(angle), qargs, cargs))
                        elif inst.name == "rx":
                            out.append((RXGate(angle), qargs, cargs))
                        else:
                            out.append((RYGate(angle), qargs, cargs))
                        changed = True
                        i += 2
                        continue

        out.append((inst, qargs, cargs))
        i += 1

    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for op, qargs, cargs in out:
        new_qc.append(op, qargs, cargs)
    return new_qc, changed


def simplify(qc: QuantumCircuit, max_iterations: int = 50) -> QuantumCircuit:
    """
    Repeatedly apply cancellation and merge rules until stable.
    """
    current = qc
    for _ in range(max_iterations):
        current, changed = _apply_rules_once(current)
        if not changed:
            break
    return current


def _insert_identity_pair(
    qc: QuantumCircuit,
    insert_after: int,
    qubit,
    axis: str,
    angle: float,
) -> QuantumCircuit:
    data = list(qc.data)
    out = []
    for idx, item in enumerate(data):
        out.append(item)
        if idx == insert_after:
            if axis == "rz":
                out.append((RZGate(angle), [qubit], []))
                out.append((RZGate(-angle), [qubit], []))
            else:
                out.append((RXGate(angle), [qubit], []))
                out.append((RXGate(-angle), [qubit], []))

    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for op, qargs, cargs in out:
        new_qc.append(op, qargs, cargs)
    return new_qc


def identity_based_refinement(
    qc: QuantumCircuit,
    fitness_fn: Callable[[QuantumCircuit], float],
    angles: Tuple[float, ...] = (np.pi / 4, np.pi / 2),
) -> QuantumCircuit:
    """
    Expand with identity pairs (RZ(a)RZ(-a), RX(a)RX(-a)) and keep only
    candidates that improve fitness after simplification.
    """
    best = qc
    best_score = fitness_fn(best)
    ops = list(qc.data)

    candidates: List[QuantumCircuit] = []
    for i, (inst, qargs, _cargs) in enumerate(ops):
        if inst.name not in {"cx", "ccx"}:
            continue
        q = qargs[0]
        for a in angles:
            candidates.append(_insert_identity_pair(qc, i, q, axis="rz", angle=a))
            candidates.append(_insert_identity_pair(qc, i, q, axis="rx", angle=a))

    for cand in candidates:
        simplified = simplify(cand)
        score = fitness_fn(simplified)
        if score > best_score:
            best = simplified
            best_score = score

    return best