"""
Genetic Algorithm for evolving action-sequences that optimise
quantum circuits.

Each *individual* is a fixed-length sequence of action indices.
Fitness = -cost of the circuit after applying the sequence.
"""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit

import config
from circuit_optimizer.actions import apply_action, num_actions
from circuit_optimizer.cost import circuit_cost, fitness as calc_fitness
from circuit_optimizer.representation import copy_circuit


class Individual:
    """One member of the GA population."""

    def __init__(
        self,
        sequence: List[int],
        fitness: float = -np.inf,
        result_circuit: Optional[QuantumCircuit] = None,
    ) -> None:
        self.sequence = sequence
        self.fitness = fitness
        self.result_circuit = result_circuit


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
        early_stop_patience: Optional[int] = None,
        early_stop_min_delta: float = 1e-9,
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
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.n_actions = num_actions()
        self.verbose = verbose

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[dict] = []   # per-generation stats

    # ── public API ─────────────────────────────────────────────

    def run(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        generation_callback: Optional[Callable[[dict], None]] = None,
    ) -> Tuple[QuantumCircuit, List[dict]]:
        """Execute the GA and return (best_circuit, history)."""
        self._init_population()
        self._evaluate_all()
        best_cost_so_far = float("inf")
        no_improve_streak = 0

        for gen in range(self.generations):
            if stop_check is not None and stop_check():
                break

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
            self.history.append(gen_stats)

            current_best_cost = float(gen_stats["best_cost"])
            if current_best_cost < (best_cost_so_far - self.early_stop_min_delta):
                best_cost_so_far = current_best_cost
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            if generation_callback is not None:
                try:
                    generation_callback(gen_stats)
                except Exception:
                    # Progress callbacks must not break optimization.
                    pass

            if self.verbose:
                print(f"  Gen {gen:3d} | best_cost={gen_stats['best_cost']:.4f}"
                      f"  avg_cost={-gen_stats['avg_fitness']:.4f}")

            if (
                self.early_stop_patience is not None
                and self.early_stop_patience > 0
                and no_improve_streak >= self.early_stop_patience
            ):
                if self.verbose:
                    print(
                        "  Early stop: best_cost did not improve for "
                        f"{self.early_stop_patience} generations."
                    )
                break

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

    def _evaluate_all(self):
        for ind in self.population:
            if ind.fitness == -np.inf:
                self._evaluate(ind)
        # Track global best
        best = max(self.population, key=lambda i: i.fitness)
        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(
                sequence=list(best.sequence),
                fitness=best.fitness,
                result_circuit=copy_circuit(best.result_circuit),
            )

    def _select_elite(self) -> List[Individual]:
        sorted_pop = sorted(self.population, key=lambda i: i.fitness, reverse=True)
        return [Individual(sequence=list(ind.sequence),
                           fitness=ind.fitness,
                           result_circuit=copy_circuit(ind.result_circuit))
                for ind in sorted_pop[:self.elite_count]]

    def _tournament_select(self) -> Individual:
        participants = random.sample(self.population, self.tournament_k)
        return max(participants, key=lambda i: i.fitness)

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
