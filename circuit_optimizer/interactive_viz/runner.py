"""Runner helpers for the Streamlit visualization app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from qiskit import QuantumCircuit

from circuit_optimizer.circuits.generators import random_redundant_circuit
from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.equivalence import check_equivalence
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm
from circuit_optimizer.hybrid_optimizer import HybridOptimizer
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.rl_agent import RLAgent


@dataclass
class OptimizationResult:
    """Container for one generation + optimization run."""

    original_circuit: QuantumCircuit
    optimized_circuit: QuantumCircuit
    original_metrics: Dict
    optimized_metrics: Dict
    method_used: str
    model_message: str
    equivalence_result: Dict = None
    ga_circuit: Optional[QuantumCircuit] = None
    ga_metrics: Optional[Dict] = None
    ga_equivalence_result: Optional[Dict] = None
    hybrid_circuit: Optional[QuantumCircuit] = None
    hybrid_metrics: Optional[Dict] = None
    hybrid_equivalence_result: Optional[Dict] = None


def _try_load_agent(model_path: str) -> tuple[Optional[RLAgent], str]:
    """Try loading an RL model for hybrid optimization."""
    agent = RLAgent(algorithm="PPO", verbose=0)
    try:
        agent.load(model_path)
        return agent, f"Loaded RL model from '{model_path}'."
    except Exception as exc:
        return None, (
            f"Could not load RL model from '{model_path}' ({exc}). "
            "Falling back to GA-only optimization."
        )


def _run_ga(
    base_circuit: QuantumCircuit,
    ga_generations: int,
    ga_pop_size: int,
) -> QuantumCircuit:
    """Run the GA optimizer and return the best circuit found."""
    ga = GeneticAlgorithm(
        circuit_fn=lambda: copy_circuit(base_circuit),
        generations=ga_generations,
        pop_size=ga_pop_size,
        verbose=False,
    )
    best_qc, _ = ga.run()
    return best_qc


def _run_hybrid(
    base_circuit: QuantumCircuit,
    rl_agent: RLAgent,
    ga_generations: int,
    ga_pop_size: int,
) -> QuantumCircuit:
    """Run the hybrid optimizer and return the best circuit found."""
    hybrid = HybridOptimizer(
        circuit_fn=lambda: copy_circuit(base_circuit),
        rl_agent=rl_agent,
        generations=ga_generations,
        pop_size=ga_pop_size,
        verbose=False,
    )
    best_qc, _ = hybrid.run()

    # Optional polish step with the learned policy.
    polished = rl_agent.optimize(best_qc)
    if circuit_cost(polished) < circuit_cost(best_qc):
        return polished
    return best_qc


def generate_and_optimize(
    n_qubits: int,
    depth: int,
    seed: Optional[int],
    ga_generations: int,
    ga_pop_size: int,
    model_path: str,
    max_attempts: int = 3,
) -> OptimizationResult:
    """Generate random circuits and optimize until gate reduction is found or attempts end."""
    rl_agent, model_message = _try_load_agent(model_path)

    best_result: Optional[OptimizationResult] = None
    best_gate_delta = 10**9

    for attempt in range(max_attempts):
        trial_seed = None if seed is None else seed + attempt
        original = random_redundant_circuit(n_qubits=n_qubits, depth=depth, seed=trial_seed)
        original_metrics = circuit_metrics(original)

        if rl_agent is not None and rl_agent.model is not None:
            ga_optimized = _run_ga(
                base_circuit=original,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
            )
            hybrid_optimized = _run_hybrid(
                base_circuit=original,
                rl_agent=rl_agent,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
            )

            ga_metrics = circuit_metrics(ga_optimized)
            hybrid_metrics = circuit_metrics(hybrid_optimized)

            if hybrid_metrics["cost"] <= ga_metrics["cost"]:
                optimized = hybrid_optimized
                optimized_metrics = hybrid_metrics
                method_used = "Hybrid (GA+RL)"
            else:
                optimized = ga_optimized
                optimized_metrics = ga_metrics
                method_used = "GA"

            ga_equivalence = check_equivalence(original, ga_optimized)
            hybrid_equivalence = check_equivalence(original, hybrid_optimized)
        else:
            optimized = _run_ga(
                base_circuit=original,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
            )
            optimized_metrics = circuit_metrics(optimized)
            method_used = "GA"
            ga_optimized = optimized
            ga_metrics = optimized_metrics
            ga_equivalence = check_equivalence(original, optimized)
            hybrid_optimized = None
            hybrid_metrics = None
            hybrid_equivalence = None

        gate_delta = optimized_metrics["total_gates"] - original_metrics["total_gates"]

        current = OptimizationResult(
            original_circuit=original,
            optimized_circuit=optimized,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            method_used=method_used,
            model_message=model_message,
            equivalence_result=check_equivalence(original, optimized),
            ga_circuit=ga_optimized,
            ga_metrics=ga_metrics,
            ga_equivalence_result=ga_equivalence,
            hybrid_circuit=hybrid_optimized,
            hybrid_metrics=hybrid_metrics,
            hybrid_equivalence_result=hybrid_equivalence,
        )

        if gate_delta < best_gate_delta:
            best_gate_delta = gate_delta
            best_result = current

        if gate_delta < 0:
            return current

    return best_result
