"""
Hybrid GA + RL optimiser.

Architecture:
  Phase 1 – GA with RL-guided mutation (global exploration)
  Phase 2 – RL greedy refinement of top-k GA results (local exploitation)

The GA runs exactly as usual, but mutations are informed by the RL
policy.  After the GA finishes, the best individuals get a second pass
of greedy local search.  This guarantees the hybrid is *at least* as
good as GA-only.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit

import config
from circuit_optimizer.actions import apply_action, num_actions
from circuit_optimizer.cost import circuit_cost, fitness as calc_fitness
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm, Individual
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.rl_agent import RLAgent


class HybridOptimizer(GeneticAlgorithm):
    """
    Two improvements over GA-only:

    1. **RL-guided mutation**: when a gene mutates, it has a chance
       of being set by the RL policy (instead of random).
    2. **Post-GA refinement**: after the GA finishes, the top-k
       circuits are further improved with greedy local search.
       This is done *outside* the GA loop so it doesn't distort
       the fitness landscape.
    """

    def __init__(
        self,
        circuit_fn: Callable[[], QuantumCircuit],
        rl_agent: RLAgent,
        rl_mutation_prob: float = config.HYBRID_RL_MUTATION_PROB,
        rl_refine_top_k: int = 5,
        rl_refine_steps: int = 25,
        **ga_kwargs,
    ):
        super().__init__(circuit_fn=circuit_fn, **ga_kwargs)
        self.rl_agent = rl_agent
        self.rl_mutation_prob = rl_mutation_prob
        self.rl_refine_top_k = rl_refine_top_k
        self.rl_refine_steps = rl_refine_steps

    # ── override: RL-guided mutation ───────────────────────────
    def _mutate(self, seq: List[int]) -> List[int]:
        """
        Point mutation with RL guidance.

        For each position that mutates:
          - with prob rl_mutation_prob  →  ask RL policy for the action
          - otherwise                  →  random action
        """
        qc = self.circuit_fn()

        for i in range(len(seq)):
            if random.random() < self.mutation_rate:
                if (random.random() < self.rl_mutation_prob
                        and self.rl_agent is not None
                        and self.rl_agent.model is not None):
                    action = self.rl_agent.suggest_action(qc)
                    seq[i] = action
                else:
                    seq[i] = random.randint(0, self.n_actions - 1)

            # Advance scratch circuit regardless
            new_qc = apply_action(qc, seq[i])
            if new_qc is not None:
                qc = new_qc

        return seq

    # ── override: add post-GA refinement ───────────────────────
    def run(self) -> Tuple[QuantumCircuit, List[dict]]:
        """
        Phase 1: Run GA with RL-guided mutations.
        Phase 2: Greedy-refine top-k results from GA.
        """
        # Phase 1 – standard GA run (with RL-guided mutation)
        best_qc, history = super().run()
        ga_cost = circuit_cost(best_qc)

        # Phase 2 – refine top-k individuals
        if self.rl_agent is not None:
            # Collect unique candidate circuits from population
            candidates = []
            seen_costs = set()
            sorted_pop = sorted(
                self.population, key=lambda i: i.fitness, reverse=True
            )
            for ind in sorted_pop:
                if ind.result_circuit is not None:
                    c = round(circuit_cost(ind.result_circuit), 6)
                    if c not in seen_costs:
                        seen_costs.add(c)
                        candidates.append(ind.result_circuit)
                if len(candidates) >= self.rl_refine_top_k:
                    break

            # Add the overall best too
            candidates.insert(0, best_qc)

            # Refine each candidate
            overall_best = best_qc
            overall_best_cost = ga_cost
            for cand in candidates:
                refined = self.rl_agent.refine(
                    cand, max_steps=self.rl_refine_steps
                )
                rc = circuit_cost(refined)
                if rc < overall_best_cost:
                    overall_best_cost = rc
                    overall_best = refined

            best_qc = overall_best

        return best_qc, history

    @staticmethod
    def full_pipeline(
        circuit_fn: Callable[[], QuantumCircuit],
        rl_timesteps: int = config.RL_TOTAL_TIMESTEPS,
        ga_generations: int = config.GA_GENERATIONS,
        ga_pop_size: int = config.GA_POPULATION_SIZE,
        verbose: bool = True,
    ) -> Tuple[QuantumCircuit, dict]:
        """
        End-to-end:  train RL → run hybrid GA → return best circuit.

        Returns
        -------
        (best_circuit, stats_dict)
        """
        if verbose:
            print("=== Phase 1: Training RL agent ===")
        agent = RLAgent(
            algorithm="PPO",
            circuit_generator=circuit_fn,
            total_timesteps=rl_timesteps,
            verbose=1 if verbose else 0,
        )
        rl_stats = agent.train()

        if verbose:
            print(f"\nRL training done. Avg improvement: "
                  f"{rl_stats['avg_improvement_pct']:.2f}%\n")
            print("=== Phase 2: Hybrid GA + RL ===")

        hybrid = HybridOptimizer(
            circuit_fn=circuit_fn,
            rl_agent=agent,
            pop_size=ga_pop_size,
            generations=ga_generations,
            verbose=verbose,
        )
        best_qc, ga_history = hybrid.run()

        stats = {
            "rl_stats": rl_stats,
            "ga_history": ga_history,
            "final_cost": circuit_cost(best_qc),
        }
        return best_qc, stats
