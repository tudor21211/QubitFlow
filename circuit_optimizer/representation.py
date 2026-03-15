"""
Circuit representation utilities.

Converts a Qiskit QuantumCircuit into feature vectors that the RL agent
can observe, and provides DAG-based helpers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit


# ── Gate categories ────────────────────────────────────────────────
SINGLE_QUBIT_GATES = {"x", "y", "z", "h", "s", "sdg", "t", "tdg",
                       "rx", "ry", "rz", "u1", "u2", "u3", "id", "sx", "sxdg"}
TWO_QUBIT_GATES = {"cx", "cz", "cy", "swap", "ecr", "rzz", "rxx", "ryy",
                    "cp", "crx", "cry", "crz"}
THREE_QUBIT_GATES = {"ccx", "cswap"}


def circuit_features(qc: QuantumCircuit) -> np.ndarray:
    """
    Extract a fixed-size numerical feature vector from a QuantumCircuit.

    Features (12-dim):
        0  - total gate count
        1  - depth
        2  - number of qubits
        3  - CNOT (CX) count
        4  - single-qubit gate count
        5  - two-qubit gate count
        6  - three+-qubit gate count
        7  - ratio CX / total gates  (0 if no gates)
        8  - ratio depth / num_qubits
        9  - max gates on any qubit
        10 - min gates on any qubit
        11 - std-dev of gates per qubit
    """
    ops = dict(qc.count_ops())
    total_gates = sum(ops.values())
    depth = qc.depth()
    n_qubits = qc.num_qubits

    cx_count = ops.get("cx", 0)
    single = sum(ops.get(g, 0) for g in SINGLE_QUBIT_GATES)
    two = sum(ops.get(g, 0) for g in TWO_QUBIT_GATES)
    three = sum(ops.get(g, 0) for g in THREE_QUBIT_GATES)

    cx_ratio = cx_count / total_gates if total_gates > 0 else 0.0
    depth_ratio = depth / n_qubits if n_qubits > 0 else 0.0

    # Per-qubit gate histogram
    dag = circuit_to_dag(qc)
    per_qubit = _gates_per_qubit(dag, n_qubits)
    max_gq = float(np.max(per_qubit)) if n_qubits > 0 else 0.0
    min_gq = float(np.min(per_qubit)) if n_qubits > 0 else 0.0
    std_gq = float(np.std(per_qubit)) if n_qubits > 0 else 0.0

    return np.array([
        total_gates, depth, n_qubits, cx_count,
        single, two, three,
        cx_ratio, depth_ratio,
        max_gq, min_gq, std_gq,
    ], dtype=np.float32)


def _gates_per_qubit(dag: DAGCircuit, n_qubits: int) -> np.ndarray:
    """Count how many gates touch each qubit."""
    counts = np.zeros(n_qubits, dtype=np.float32)
    for node in dag.op_nodes():
        for qarg in node.qargs:
            idx = dag.find_bit(qarg).index
            counts[idx] += 1
    return counts


def dag_depth(dag: DAGCircuit) -> int:
    """Depth of a DAGCircuit (longest path through operations)."""
    return dag.depth()


def dag_gate_count(dag: DAGCircuit) -> int:
    """Total number of operation nodes in the DAG."""
    return dag.size()


def dag_cx_count(dag: DAGCircuit) -> int:
    """Count CX gates in a DAG."""
    return sum(1 for nd in dag.op_nodes() if nd.op.name == "cx")


def copy_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a deep copy of the circuit."""
    return qc.copy()
