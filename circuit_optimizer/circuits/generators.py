"""
Test circuit generators.

Produces quantum circuits of varying size and type for benchmarking
the optimiser.  Circuits intentionally contain redundancies so the
optimiser has something to improve.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    QFT,
    RealAmplitudes,
)


def random_redundant_circuit(
    n_qubits: int = 4,
    depth: int = 20,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Build a circuit with random gates **and** deliberate redundancies
    (double-H, CX-CX, etc.) so the optimiser can find improvements.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)

    single_gates = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
    rotation_gates = ["rx", "ry", "rz"]

    for _ in range(depth):
        r = rng.random()
        if r < 0.30:
            # Random single-qubit gate
            gate = rng.choice(single_gates)
            q = rng.randint(0, n_qubits - 1)
            getattr(qc, gate)(q)
        elif r < 0.50:
            # Rotation gate
            gate = rng.choice(rotation_gates)
            angle = rng.uniform(-np.pi, np.pi)
            q = rng.randint(0, n_qubits - 1)
            getattr(qc, gate)(angle, q)
        elif r < 0.70:
            # CX gate
            q0, q1 = rng.sample(range(n_qubits), 2)
            qc.cx(q0, q1)
        elif r < 0.80:
            # Deliberate H-H redundancy
            q = rng.randint(0, n_qubits - 1)
            qc.h(q)
            qc.h(q)
        elif r < 0.90:
            # Deliberate CX-CX redundancy
            if n_qubits >= 2:
                q0, q1 = rng.sample(range(n_qubits), 2)
                qc.cx(q0, q1)
                qc.cx(q0, q1)
        else:
            # H-X-H sandwich (= Z)
            q = rng.randint(0, n_qubits - 1)
            qc.h(q)
            qc.x(q)
            qc.h(q)

    return qc


def qft_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """QFT circuit (well-structured, hard to compress much)."""
    return QFT(n_qubits).decompose()


def variational_circuit(n_qubits: int = 4, reps: int = 2,
                        seed: int = 42) -> QuantumCircuit:
    """Parameterised variational ansatz with random parameter values."""
    rng = np.random.default_rng(seed)
    ansatz = EfficientSU2(n_qubits, reps=reps)
    params = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    bound = ansatz.assign_parameters(params)
    return bound.decompose()


def grover_like_circuit(n_qubits: int = 3) -> QuantumCircuit:
    """Simple Grover-style circuit with oracle + diffusion."""
    qc = QuantumCircuit(n_qubits)
    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)
    # Fake oracle (CZ on last qubit controlled by previous)
    for q in range(n_qubits - 1):
        qc.cx(q, n_qubits - 1)
    qc.z(n_qubits - 1)
    for q in range(n_qubits - 1):
        qc.cx(q, n_qubits - 1)
    # Diffusion
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    # Multi-controlled Z using CX chain
    for q in range(n_qubits - 1):
        qc.cx(q, n_qubits - 1)
    qc.z(n_qubits - 1)
    for q in range(n_qubits - 1):
        qc.cx(q, n_qubits - 1)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)
    return qc


def all_test_circuits() -> List[dict]:
    """
    Return a list of ``{name, circuit}`` dicts for benchmarking.
    """
    circuits = [
        {"name": "random_4q_d20", "circuit": random_redundant_circuit(4, 20, seed=0)},
        {"name": "random_4q_d40", "circuit": random_redundant_circuit(4, 40, seed=1)},
        {"name": "random_6q_d30", "circuit": random_redundant_circuit(6, 30, seed=2)},
        {"name": "qft_4", "circuit": qft_circuit(4)},
        {"name": "qft_6", "circuit": qft_circuit(6)},
        {"name": "variational_4q", "circuit": variational_circuit(4, 2, seed=42)},
        {"name": "grover_3q", "circuit": grover_like_circuit(3)},
        {"name": "grover_4q", "circuit": grover_like_circuit(4)},
    ]
    return circuits


def make_generator(n_qubits: int = 4, depth: int = 20) -> callable:
    """Return a zero-argument callable that generates random circuits."""
    def _gen():
        return random_redundant_circuit(n_qubits, depth, seed=None)
    return _gen
