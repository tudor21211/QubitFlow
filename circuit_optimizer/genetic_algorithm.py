"""
Genetic Algorithm for evolving action-sequences that optimise
quantum circuits.

Each *individual* is a fixed-length sequence of action indices.
Fitness = -cost of the circuit after applying the sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit

import config
from circuit_optimizer.actions import apply_action, num_actions
from circuit_optimizer.cost import circuit_cost, fitness as calc_fitness, circuit_objectives
from circuit_optimizer.representation import copy_circuit


@dataclass
class Individual:
    """One member of the GA population."""
    sequence: List[int]
    fitness: float = -np.inf
    result_circuit: Optional[QuantumCircuit] = None
    objectives: dict = field(default_factory=dict)
    pareto_rank: int = 10**9
    crowding_distance: float = 0.0


# ───────────────────────────────────────────────────────────────────
#  Core GA
# ───────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Evolves sequences of rewrite-rule indices to minimise circuit cost.

    Parameters
    ----------
    circuit_fn : callable() -> QuantumCircuit
        Factory that produces a fresh copy of the circuit to optimise.
    pop_size, generations, etc. : see config.py
    """

    def __init__(
        self,
        circuit_fn: Callable[[], QuantumCircuit],
        pop_size: int = config.GA_POPULATION_SIZE,
        generations: int = config.GA_GENERATIONS,
        elite_count: int = config.GA_ELITE_COUNT,
        tournament_k: int = config.GA_TOURNAMENT_SIZE,
        crossover_rate: float = config.GA_CROSSOVER_RATE,
        mutation_rate: float = config.GA_MUTATION_RATE,
        seq_length: int = config.GA_SEQUENCE_LENGTH,
        verbose: bool = True,
    ):
        self.circuit_fn = circuit_fn
        self.pop_size = pop_size
        self.generations = generations
        self.elite_count = elite_count
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seq_length = seq_length
        self.n_actions = num_actions()
        self.verbose = verbose
        self.selection_mode = getattr(config, "GA_SELECTION_MODE", "weighted").lower()
        self.pareto_objectives = list(getattr(config, "GA_PARETO_OBJECTIVES", [
            "depth", "cx_count", "estimated_error"
        ]))

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[dict] = []   # per-generation stats

    # ── public API ─────────────────────────────────────────────

    def run(self) -> Tuple[QuantumCircuit, List[dict]]:
        """Execute the GA and return (best_circuit, history)."""
        self._init_population()
        self._evaluate_all()

        for gen in range(self.generations):
            new_pop: List[Individual] = []

            # Elitism
            elite = self._select_elite()
            new_pop.extend(elite)

            # Fill rest via selection + crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()

                if random.random() < self.crossover_rate:
                    c1_seq, c2_seq = self._crossover(p1.sequence, p2.sequence)
                else:
                    c1_seq = list(p1.sequence)
                    c2_seq = list(p2.sequence)

                c1_seq = self._mutate(c1_seq)
                c2_seq = self._mutate(c2_seq)

                new_pop.append(Individual(sequence=c1_seq))
                if len(new_pop) < self.pop_size:
                    new_pop.append(Individual(sequence=c2_seq))

            self.population = new_pop[:self.pop_size]
            self._evaluate_all()

            # Stats
            fits = [ind.fitness for ind in self.population]
            gen_stats = {
                "generation": gen,
                "best_fitness": max(fits),
                "avg_fitness": np.mean(fits),
                "worst_fitness": min(fits),
                "best_cost": -max(fits),
            }
            if self.selection_mode == "pareto":
                gen_stats["pareto_front_size"] = sum(
                    1 for ind in self.population if ind.pareto_rank == 0
                )
            self.history.append(gen_stats)

            if self.verbose:
                print(f"  Gen {gen:3d} | best_cost={gen_stats['best_cost']:.4f}"
                      f"  avg_cost={-gen_stats['avg_fitness']:.4f}")

        # Return result
        return copy_circuit(self.best_individual.result_circuit), self.history

    # ── internal ───────────────────────────────────────────────

    def _init_population(self):
        self.population = [
            Individual(sequence=[random.randint(0, self.n_actions - 1)
                                 for _ in range(self.seq_length)])
            for _ in range(self.pop_size)
        ]

    def _evaluate(self, ind: Individual):
        """Replay action sequence on a fresh circuit, compute fitness."""
        qc = self.circuit_fn()
        for action_idx in ind.sequence:
            new_qc = apply_action(qc, action_idx)
            if new_qc is not None:
                qc = new_qc
        ind.result_circuit = qc
        ind.fitness = calc_fitness(qc)
        ind.objectives = circuit_objectives(qc)

    def _evaluate_all(self):
        for ind in self.population:
            if ind.fitness == -np.inf:
                self._evaluate(ind)

        if self.selection_mode == "pareto":
            self._assign_pareto_metadata(self.population)

        # Track global best
        best = max(self.population, key=lambda i: i.fitness)
        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(
                sequence=list(best.sequence),
                fitness=best.fitness,
                result_circuit=copy_circuit(best.result_circuit),
            )

    def _select_elite(self) -> List[Individual]:
        if self.selection_mode == "pareto":
            sorted_pop = sorted(
                self.population,
                key=lambda i: (i.pareto_rank, -i.crowding_distance, -i.fitness),
            )
        else:
            sorted_pop = sorted(self.population, key=lambda i: i.fitness, reverse=True)
        return [Individual(sequence=list(ind.sequence),
                           fitness=ind.fitness,
                           result_circuit=copy_circuit(ind.result_circuit),
                           objectives=dict(ind.objectives),
                           pareto_rank=ind.pareto_rank,
                           crowding_distance=ind.crowding_distance)
                for ind in sorted_pop[:self.elite_count]]

    def _tournament_select(self) -> Individual:
        participants = random.sample(self.population, self.tournament_k)
        if self.selection_mode == "pareto":
            participants = sorted(
                participants,
                key=lambda i: (i.pareto_rank, -i.crowding_distance, -i.fitness),
            )
            return participants[0]
        return max(participants, key=lambda i: i.fitness)

    # ── Pareto helpers ──────────────────────────────────────────
    def _assign_pareto_metadata(self, population: List[Individual]):
        fronts = self._nondominated_sort(population)
        for rank, front in enumerate(fronts):
            for ind in front:
                ind.pareto_rank = rank
            self._assign_crowding_distance(front)

    def _nondominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        dominates = {id(ind): [] for ind in population}
        dominated_count = {id(ind): 0 for ind in population}

        for p in population:
            for q in population:
                if p is q:
                    continue
                if self._dominates(p, q):
                    dominates[id(p)].append(q)
                elif self._dominates(q, p):
                    dominated_count[id(p)] += 1

        first_front = [ind for ind in population if dominated_count[id(ind)] == 0]
        fronts: List[List[Individual]] = []
        current = first_front
        while current:
            fronts.append(current)
            next_front: List[Individual] = []
            for p in current:
                for q in dominates[id(p)]:
                    dominated_count[id(q)] -= 1
                    if dominated_count[id(q)] == 0:
                        next_front.append(q)
            current = next_front
        return fronts

    def _dominates(self, p: Individual, q: Individual) -> bool:
        keys = self.pareto_objectives
        p_vals = [float(p.objectives.get(k, np.inf)) for k in keys]
        q_vals = [float(q.objectives.get(k, np.inf)) for k in keys]

        not_worse = all(pv <= qv for pv, qv in zip(p_vals, q_vals))
        strictly_better = any(pv < qv for pv, qv in zip(p_vals, q_vals))
        return not_worse and strictly_better

    def _assign_crowding_distance(self, front: List[Individual]):
        if not front:
            return
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for key in self.pareto_objectives:
            front.sort(key=lambda i: float(i.objectives.get(key, np.inf)))
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            vmin = float(front[0].objectives.get(key, np.inf))
            vmax = float(front[-1].objectives.get(key, np.inf))
            denom = vmax - vmin
            if denom <= 0.0:
                continue

            for idx in range(1, len(front) - 1):
                if np.isinf(front[idx].crowding_distance):
                    continue
                prev_v = float(front[idx - 1].objectives.get(key, np.inf))
                next_v = float(front[idx + 1].objectives.get(key, np.inf))
                front[idx].crowding_distance += (next_v - prev_v) / denom

    @staticmethod
    def _crossover(s1: List[int], s2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover."""
        point = random.randint(1, len(s1) - 1)
        c1 = s1[:point] + s2[point:]
        c2 = s2[:point] + s1[point:]
        return c1, c2

    def _mutate(self, seq: List[int]) -> List[int]:
        """Point mutation: each gene has mutation_rate chance of flipping."""
        for i in range(len(seq)):
            if random.random() < self.mutation_rate:
                seq[i] = random.randint(0, self.n_actions - 1)
        return seq
