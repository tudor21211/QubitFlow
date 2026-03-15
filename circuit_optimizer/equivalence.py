"""
Equivalence checking for quantum circuits.

Two circuits are functionally equivalent if they implement the same
unitary transformation (up to global phase, which is physically
unobservable).

Two strategies are provided:

  1. Unitary comparison  (exact, O(4^n) memory) – practical up to ~8 qubits.
     Uses qiskit.quantum_info.Operator, which builds the full 2^n x 2^n
     unitary matrix and compares element-wise.  Operator.equiv() handles
     global phase automatically.

  2. Statevector sampling  (probabilistic, O(2^n) memory) – for larger
     circuits. Runs both circuits on the |0...0> computational basis state
     plus a set of random Haar-random input statevectors and computes the
     state fidelity between the outputs.  If every fidelity is ≥ 1-tol the
     circuits are declared equivalent with high confidence.

check_equivalence()  dispatches to strategy 1 for ≤ max_unitary_qubits
qubits, otherwise falls back to strategy 2.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, state_fidelity, random_statevector


# ── Strategy 1: exact unitary comparison ─────────────────────────


def check_unitary_equivalence(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    atol: float = 1e-6,
) -> dict:
    """
    Compare two circuits by their full unitary matrices.

    Parameters
    ----------
    qc1, qc2 : QuantumCircuit
        Must have the same number of qubits (no measurements allowed).
    atol : float
        Absolute tolerance for element-wise comparison.

    Returns
    -------
    dict with keys:
        is_equivalent  : bool
        max_abs_diff   : float  – max element-wise absolute difference
                                  after removing global phase
        method         : "unitary"
        error          : str or None  – set if an exception occurred
    """
    if qc1.num_qubits != qc2.num_qubits:
        return {
            "is_equivalent": False,
            "max_abs_diff": float("inf"),
            "method": "unitary",
            "error": (
                f"Qubit count mismatch: {qc1.num_qubits} vs {qc2.num_qubits}"
            ),
        }

    try:
        # Remove any measurements so Operator can be computed
        c1 = _remove_measurements(qc1)
        c2 = _remove_measurements(qc2)

        op1 = Operator(c1)
        op2 = Operator(c2)

        # Operator.equiv() checks up to global phase
        is_equiv = op1.equiv(op2, rtol=0, atol=atol)

        # Compute the element-wise difference after aligning global phase.
        # The global phase that minimises ||U1 - e^{i*phi} U2|| is found by
        # maximising Re(trace(U1† U2)), giving phi = angle(trace(U1† U2)).
        mat1 = op1.data
        mat2 = op2.data
        overlap = np.trace(mat1.conj().T @ mat2)
        # Remove global phase from mat2
        if abs(overlap) > 1e-12:
            phase = overlap / abs(overlap)
            mat2_aligned = mat2 * np.conj(phase)
        else:
            mat2_aligned = mat2
        max_diff = float(np.max(np.abs(mat1 - mat2_aligned)))

        return {
            "is_equivalent": bool(is_equiv),
            "max_abs_diff": max_diff,
            "method": "unitary",
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "is_equivalent": False,
            "max_abs_diff": float("inf"),
            "method": "unitary",
            "error": str(exc),
        }


# ── Strategy 2: statevector sampling ─────────────────────────────


def check_statevector_equivalence(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    n_tests: int = 8,
    atol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """
    Compare circuits by simulating several input statevectors.

    Tests:
      - The all-zeros computational basis state |0...0>
      - n_tests - 1 Haar-random input states

    For each input |ψ>, applies both circuits and computes the state
    fidelity F = |<φ1|φ2>|^2 between the outputs.  Perfect equivalence
    (up to global phase) gives F = 1.

    Parameters
    ----------
    qc1, qc2 : QuantumCircuit
    n_tests   : int  – total number of input states to test
    atol      : float – a fidelity error > atol flags non-equivalence
    seed      : int   – RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        is_equivalent    : bool
        min_fidelity     : float
        max_fidelity_error : float  –  max(1 - fidelity) over all tests
        fidelities       : list[float]
        method           : "statevector"
        error            : str or None
    """
    if qc1.num_qubits != qc2.num_qubits:
        return {
            "is_equivalent": False,
            "min_fidelity": 0.0,
            "max_fidelity_error": 1.0,
            "fidelities": [],
            "method": "statevector",
            "error": (
                f"Qubit count mismatch: {qc1.num_qubits} vs {qc2.num_qubits}"
            ),
        }

    try:
        c1 = _remove_measurements(qc1)
        c2 = _remove_measurements(qc2)
        n = c1.num_qubits
        rng = np.random.default_rng(seed)

        fidelities: list[float] = []

        for i in range(n_tests):
            if i == 0:
                # Computational basis zero state
                psi = Statevector.from_int(0, 2**n)
            else:
                # Haar-random input state
                psi = random_statevector(2**n, seed=int(rng.integers(0, 2**31)))

            out1 = psi.evolve(c1)
            out2 = psi.evolve(c2)
            fid = float(state_fidelity(out1, out2, validate=False))
            fidelities.append(fid)

        max_err = float(max(1.0 - f for f in fidelities))
        is_equiv = max_err <= atol

        return {
            "is_equivalent": is_equiv,
            "min_fidelity": float(min(fidelities)),
            "max_fidelity_error": max_err,
            "fidelities": fidelities,
            "method": "statevector",
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "is_equivalent": False,
            "min_fidelity": 0.0,
            "max_fidelity_error": 1.0,
            "fidelities": [],
            "method": "statevector",
            "error": str(exc),
        }


# ── Unified dispatcher ────────────────────────────────────────────


def check_equivalence(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    max_unitary_qubits: int = 8,
    atol: float = 1e-6,
) -> dict:
    """
    Check whether qc1 and qc2 implement the same unitary transformation.

    Dispatches to:
      - check_unitary_equivalence  for ≤ max_unitary_qubits qubits (exact)
      - check_statevector_equivalence  otherwise (probabilistic)

    Returns
    -------
    dict – see check_unitary_equivalence / check_statevector_equivalence
    """
    n = max(qc1.num_qubits, qc2.num_qubits)
    if n <= max_unitary_qubits:
        return check_unitary_equivalence(qc1, qc2, atol=atol)
    return check_statevector_equivalence(qc1, qc2, atol=atol)


def format_equivalence_result(result: dict, label: str = "") -> str:
    """Return a human-readable single-line summary of an equivalence result."""
    prefix = f"[{label}] " if label else ""
    if result.get("error"):
        return f"{prefix}ERROR: {result['error']}"

    status = "PASS" if result["is_equivalent"] else "FAIL"
    method = result["method"]

    if method == "unitary":
        err = result["max_abs_diff"]
        return f"{prefix}{status}  (unitary, max_diff={err:.2e})"
    else:
        err = result["max_fidelity_error"]
        min_f = result["min_fidelity"]
        return (
            f"{prefix}{status}  "
            f"(statevector, min_fidelity={min_f:.8f}, max_err={err:.2e})"
        )


# ── Helpers ────────────────────────────────────────────────────────


def _remove_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of the circuit with all measurements and resets removed."""
    clean = qc.copy()
    clean.remove_final_measurements(inplace=True)
    return clean
