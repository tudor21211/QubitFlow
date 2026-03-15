"""
Rewrite rules (actions) that the RL agent / GA can apply to a circuit.

Every action is a function  DAGCircuit -> DAGCircuit | None.
Returning *None* means the rule was not applicable.
All rules preserve functional equivalence.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library import HGate, CXGate, XGate, SwapGate
import numpy as np


# ───────────────────────────────────────────────────────────────────
#  Rule registry
# ───────────────────────────────────────────────────────────────────

RewriteRule = Callable[[QuantumCircuit], Optional[QuantumCircuit]]

_RULES: List[RewriteRule] = []


def register(fn: RewriteRule) -> RewriteRule:
    """Decorator that adds a rule to the global list."""
    _RULES.append(fn)
    return fn


def get_rules() -> List[RewriteRule]:
    return list(_RULES)


def num_actions() -> int:
    return len(_RULES)


def apply_action(qc: QuantumCircuit, action_idx: int) -> Optional[QuantumCircuit]:
    """Apply the action at *action_idx*.  Returns None if not applicable."""
    rules = get_rules()
    if 0 <= action_idx < len(rules):
        return rules[action_idx](qc)
    return None


# ───────────────────────────────────────────────────────────────────
#  1. Cancel adjacent inverse gate pairs
# ───────────────────────────────────────────────────────────────────

_INVERSE_PAIRS = {
    ("x", "x"), ("y", "y"), ("z", "z"),
    ("h", "h"), ("s", "sdg"), ("sdg", "s"),
    ("t", "tdg"), ("tdg", "t"),
    ("cx", "cx"), ("cz", "cz"), ("swap", "swap"),
    ("sx", "sxdg"), ("sxdg", "sx"),
}


@register
def cancel_inverse_pairs(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Remove the first adjacent pair of mutually-inverse gates."""
    dag = circuit_to_dag(qc)
    gate_list = list(dag.topological_op_nodes())
    changed = False

    for node in list(dag.topological_op_nodes()):
        if node not in dag.op_nodes():
            continue
        successors = _direct_successors(dag, node)
        for succ in successors:
            pair = (node.op.name, succ.op.name)
            reverse_pair = (succ.op.name, node.op.name)
            if (pair in _INVERSE_PAIRS or reverse_pair in _INVERSE_PAIRS):
                if _same_qargs(dag, node, succ):
                    # Conservative safety check: only cancel if the two gates are
                    # consecutive in the global topological op ordering.
                    # This avoids canceling across intermediate operations.
                    idx_node = gate_list.index(node)
                    idx_succ = gate_list.index(succ)
                    if idx_succ != idx_node + 1:
                        continue
                    dag.remove_op_node(succ)
                    dag.remove_op_node(node)
                    changed = True
                    break
        if changed:
            break

    return dag_to_circuit(dag) if changed else None


# ───────────────────────────────────────────────────────────────────
#  2. Commute CX gates that share a control or target
# ───────────────────────────────────────────────────────────────────

@register
def commute_cx_shared_control(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """If two CX gates share the same control qubit, swap their order
    (they commute). Pick the first such opportunity."""
    dag = circuit_to_dag(qc)
    cx_nodes = [n for n in dag.topological_op_nodes() if n.op.name == "cx"]

    for i, a in enumerate(cx_nodes):
        for b in cx_nodes[i + 1:]:
            a_ctrl = dag.find_bit(a.qargs[0]).index
            b_ctrl = dag.find_bit(b.qargs[0]).index
            a_targ = dag.find_bit(a.qargs[1]).index
            b_targ = dag.find_bit(b.qargs[1]).index
            if a_ctrl == b_ctrl and a_targ != b_targ:
                if _are_adjacent(dag, a, b):
                    # Swap by rebuilding
                    return _swap_two_nodes(qc, dag, a, b)
    return None


@register
def commute_cx_shared_target(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """If two CX gates share the same target qubit, swap their order."""
    dag = circuit_to_dag(qc)
    cx_nodes = [n for n in dag.topological_op_nodes() if n.op.name == "cx"]

    for i, a in enumerate(cx_nodes):
        for b in cx_nodes[i + 1:]:
            a_targ = dag.find_bit(a.qargs[1]).index
            b_targ = dag.find_bit(b.qargs[1]).index
            a_ctrl = dag.find_bit(a.qargs[0]).index
            b_ctrl = dag.find_bit(b.qargs[0]).index
            if a_targ == b_targ and a_ctrl != b_ctrl:
                if _are_adjacent(dag, a, b):
                    return _swap_two_nodes(qc, dag, a, b)
    return None


# ───────────────────────────────────────────────────────────────────
#  3. HXH = Z  (Hadamard sandwich)
# ───────────────────────────────────────────────────────────────────

@register
def hadamard_sandwich_x_to_z(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Replace H-X-H with Z on the same qubit."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node.op.name != "h":
            continue
        succs = _direct_successors_on_qubit(dag, node, dag.find_bit(node.qargs[0]).index)
        for mid in succs:
            if mid.op.name != "x":
                continue
            succs2 = _direct_successors_on_qubit(dag, mid, dag.find_bit(mid.qargs[0]).index)
            for last in succs2:
                if last.op.name == "h" and _same_qubit(dag, node, last):
                    # Replace the three with Z
                    from qiskit.circuit.library import ZGate
                    dag.substitute_node(mid, ZGate(), inplace=True)
                    dag.remove_op_node(node)
                    dag.remove_op_node(last)
                    return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  4. HZH = X
# ───────────────────────────────────────────────────────────────────

@register
def hadamard_sandwich_z_to_x(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Replace H-Z-H with X on the same qubit."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node.op.name != "h":
            continue
        succs = _direct_successors_on_qubit(dag, node, dag.find_bit(node.qargs[0]).index)
        for mid in succs:
            if mid.op.name != "z":
                continue
            succs2 = _direct_successors_on_qubit(dag, mid, dag.find_bit(mid.qargs[0]).index)
            for last in succs2:
                if last.op.name == "h" and _same_qubit(dag, node, last):
                    dag.substitute_node(mid, XGate(), inplace=True)
                    dag.remove_op_node(node)
                    dag.remove_op_node(last)
                    return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  5. Cancel double-H (HH = I)
# ───────────────────────────────────────────────────────────────────

@register
def cancel_double_hadamard(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Remove two adjacent Hadamard gates on the same qubit."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node not in dag.op_nodes():
            continue
        if node.op.name != "h":
            continue
        q_idx = dag.find_bit(node.qargs[0]).index
        succs = _direct_successors_on_qubit(dag, node, q_idx)
        for s in succs:
            if s.op.name == "h" and _same_qubit(dag, node, s):
                dag.remove_op_node(s)
                dag.remove_op_node(node)
                return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  6. SWAP decomposition (SWAP = 3 CX)
# ───────────────────────────────────────────────────────────────────

@register
def decompose_swap(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Replace the first SWAP gate with 3 CX gates."""
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if node.op.name == "swap":
            q0 = node.qargs[0]
            q1 = node.qargs[1]
            mini = QuantumCircuit(2)
            mini.cx(0, 1)
            mini.cx(1, 0)
            mini.cx(0, 1)
            dag.substitute_node_with_dag(node, circuit_to_dag(mini))
            return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  7. Merge rotations:  Rz(a) Rz(b) = Rz(a+b)
# ───────────────────────────────────────────────────────────────────

@register
def merge_rz_rotations(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Merge two adjacent Rz gates on the same qubit into one."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node not in dag.op_nodes():
            continue
        if node.op.name != "rz":
            continue
        q_idx = dag.find_bit(node.qargs[0]).index
        succs = _direct_successors_on_qubit(dag, node, q_idx)
        for s in succs:
            if s.op.name == "rz" and _same_qubit(dag, node, s):
                from qiskit.circuit.library import RZGate
                angle_sum = float(node.op.params[0]) + float(s.op.params[0])
                dag.substitute_node(node, RZGate(angle_sum), inplace=True)
                dag.remove_op_node(s)
                return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  8. Merge Rx rotations
# ───────────────────────────────────────────────────────────────────

@register
def merge_rx_rotations(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Merge two adjacent Rx gates on the same qubit."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node not in dag.op_nodes():
            continue
        if node.op.name != "rx":
            continue
        q_idx = dag.find_bit(node.qargs[0]).index
        succs = _direct_successors_on_qubit(dag, node, q_idx)
        for s in succs:
            if s.op.name == "rx" and _same_qubit(dag, node, s):
                from qiskit.circuit.library import RXGate
                angle_sum = float(node.op.params[0]) + float(s.op.params[0])
                dag.substitute_node(node, RXGate(angle_sum), inplace=True)
                dag.remove_op_node(s)
                return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  9. Remove identity rotations (angle ≈ 0)
# ───────────────────────────────────────────────────────────────────

@register
def remove_identity_rotations(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Remove rotation gates whose angle is effectively zero."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node.op.name in ("rz", "rx", "ry", "u1"):
            angle = float(node.op.params[0])
            if abs(angle) < 1e-10 or abs(abs(angle) - 2 * np.pi) < 1e-10:
                dag.remove_op_node(node)
                return dag_to_circuit(dag)
    return None


# ───────────────────────────────────────────────────────────────────
#  10. Commute single-qubit gate past CX control
# ───────────────────────────────────────────────────────────────────

_Z_FAMILY = {"z", "s", "sdg", "t", "tdg", "rz", "u1"}


@register
def commute_z_past_cx_control(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Z-family gates commute with CX on the control qubit.
    Move such a gate past a CX to potentially enable further
    cancellation."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node.op.name not in _Z_FAMILY:
            continue
        q_idx = dag.find_bit(node.qargs[0]).index
        succs = _direct_successors_on_qubit(dag, node, q_idx)
        for s in succs:
            if s.op.name == "cx":
                ctrl_idx = dag.find_bit(s.qargs[0]).index
                if ctrl_idx == q_idx:
                    return _swap_two_nodes(qc, dag, node, s)
    return None


# ───────────────────────────────────────────────────────────────────
#  11. Commute X-family gate past CX target
# ───────────────────────────────────────────────────────────────────

_X_FAMILY = {"x", "sx", "sxdg", "rx"}


@register
def commute_x_past_cx_target(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """X-family gates commute with CX on the target qubit."""
    dag = circuit_to_dag(qc)
    for node in list(dag.topological_op_nodes()):
        if node.op.name not in _X_FAMILY:
            continue
        q_idx = dag.find_bit(node.qargs[0]).index
        succs = _direct_successors_on_qubit(dag, node, q_idx)
        for s in succs:
            if s.op.name == "cx":
                targ_idx = dag.find_bit(s.qargs[1]).index
                if targ_idx == q_idx:
                    return _swap_two_nodes(qc, dag, node, s)
    return None


# ───────────────────────────────────────────────────────────────────
#  Helper utilities
# ───────────────────────────────────────────────────────────────────

def _direct_successors(dag: DAGCircuit, node: DAGOpNode):
    """Return DAGOpNode successors that are directly connected."""
    result = []
    for succ in dag.successors(node):
        if isinstance(succ, DAGOpNode):
            result.append(succ)
    return result


def _direct_successors_on_qubit(dag: DAGCircuit, node: DAGOpNode, qubit_idx: int):
    """Successors that act on the given qubit index."""
    result = []
    for succ in dag.successors(node):
        if isinstance(succ, DAGOpNode):
            for qa in succ.qargs:
                if dag.find_bit(qa).index == qubit_idx:
                    result.append(succ)
                    break
    return result


def _same_qargs(dag: DAGCircuit, a: DAGOpNode, b: DAGOpNode) -> bool:
    """Check if two nodes act on exactly the same qubits in the same order."""
    if len(a.qargs) != len(b.qargs):
        return False
    for qa, qb in zip(a.qargs, b.qargs):
        if dag.find_bit(qa).index != dag.find_bit(qb).index:
            return False
    return True


def _same_qubit(dag: DAGCircuit, a: DAGOpNode, b: DAGOpNode) -> bool:
    """Both single-qubit nodes on the same qubit?"""
    if len(a.qargs) != 1 or len(b.qargs) != 1:
        return False
    return dag.find_bit(a.qargs[0]).index == dag.find_bit(b.qargs[0]).index


def _are_adjacent(dag: DAGCircuit, a: DAGOpNode, b: DAGOpNode) -> bool:
    """Check if b is a direct successor of a on any shared qubit."""
    return b in _direct_successors(dag, a)


def _swap_two_nodes(qc: QuantumCircuit, dag: DAGCircuit,
                    a: DAGOpNode, b: DAGOpNode) -> Optional[QuantumCircuit]:
    """Rebuild the circuit with two nodes swapped.
    Simple approach: rebuild from gate list with swapped positions."""
    gate_list = list(dag.topological_op_nodes())
    idx_a = gate_list.index(a)
    idx_b = gate_list.index(b)

    # Safety first: only allow swapping consecutive op-nodes.
    # Swapping non-consecutive nodes can move gates across unrelated
    # operations and break circuit semantics.
    if idx_b != idx_a + 1:
        return None

    gate_list[idx_a], gate_list[idx_b] = gate_list[idx_b], gate_list[idx_a]

    new_qc = QuantumCircuit(qc.num_qubits)
    for node in gate_list:
        qubit_indices = [dag.find_bit(q).index for q in node.qargs]
        new_qc.append(node.op, qubit_indices)
    return new_qc
