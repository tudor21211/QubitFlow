"""Benchmarking harness for GA/RL/Qiskit comparisons."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit

import config
from circuit_optimizer.baseline import transpile_baseline
from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm
from circuit_optimizer.rl_agent import RLAgent
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.equivalence import check_equivalence, format_equivalence_result
from circuit_optimizer.simplify import simplify, identity_based_refinement
from circuit_optimizer.cost import fitness


def benchmark_single(
    qc: QuantumCircuit,
    name: str = "circuit",
    rl_agent: Optional[RLAgent] = None,
    ga_generations: int = config.GA_GENERATIONS,
    ga_pop_size: int = config.GA_POPULATION_SIZE,
    runs: int = config.BENCHMARK_RUNS,
    verbose: bool = True,
) -> Dict:
    """
    Run all methods on a single circuit and return results dict.
    """
    results: Dict = {"name": name}

    # ── 1. Original ────────────────────────────────────────────
    orig_metrics = circuit_metrics(qc)
    results["original"] = orig_metrics
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Circuit: {name}")
        print(f"  Original cost: {orig_metrics['cost']:.4f}  "
              f"depth={orig_metrics['depth']}  cx={orig_metrics['cx_count']}  "
              f"gates={orig_metrics['total_gates']}")

    # ── 2. Qiskit baseline / direct Qiskit optimisation ────────
    t0 = time.time()
    base_qc = transpile_baseline(qc, optimization_level=3)
    base_time = time.time() - t0
    base_m = circuit_metrics(base_qc)
    base_m["time_s"] = base_time
    base_m["equivalence"] = check_equivalence(qc, base_qc)
    results["qiskit"] = base_m
    if verbose:
        eq_line = format_equivalence_result(base_m["equivalence"])
        print(f"  Qiskit cost:    {base_m['cost']:.4f}  "
              f"depth={base_m['depth']}  cx={base_m['cx_count']}  "
              f"time={base_time:.2f}s  equiv={eq_line}")

    # ── 3. GA only ─────────────────────────────────────────────
    circuit_fn = lambda _qc=qc: copy_circuit(_qc)
    ga_costs = []
    ga_times = []
    best_ga_qc = None
    best_ga_cost = np.inf

    for r in range(runs):
        t0 = time.time()
        ga = GeneticAlgorithm(
            circuit_fn=circuit_fn,
            generations=ga_generations,
            pop_size=ga_pop_size,
            verbose=False,
        )
        ga_qc, _ = ga.run()
        elapsed = time.time() - t0
        c = circuit_cost(ga_qc)
        ga_costs.append(c)
        ga_times.append(elapsed)
        if c < best_ga_cost:
            best_ga_cost = c
            best_ga_qc = ga_qc

    ga_m = circuit_metrics(best_ga_qc)
    ga_m["avg_cost"] = float(np.mean(ga_costs))
    ga_m["std_cost"] = float(np.std(ga_costs))
    ga_m["avg_time_s"] = float(np.mean(ga_times))
    ga_m["equivalence"] = check_equivalence(qc, best_ga_qc)
    results["ga"] = ga_m
    if verbose:
        eq_line = format_equivalence_result(ga_m["equivalence"])
        print(f"  GA cost:        {ga_m['cost']:.4f}  "
              f"(avg={ga_m['avg_cost']:.4f} +/- {ga_m['std_cost']:.4f})  "
              f"time={ga_m['avg_time_s']:.2f}s  equiv={eq_line}")

    # ── 4. RL only ──────────────────────────────────────────────
    if rl_agent is not None and rl_agent.model is not None:
        rl_costs = []
        rl_times = []
        best_rl_qc = None
        best_rl_cost = np.inf

        for r in range(runs):
            t0 = time.time()
            h_qc = rl_agent.optimize(copy_circuit(qc))
            elapsed = time.time() - t0
            c = circuit_cost(h_qc)
            rl_costs.append(c)
            rl_times.append(elapsed)
            if c < best_rl_cost:
                best_rl_cost = c
                best_rl_qc = h_qc

        rl_m = circuit_metrics(best_rl_qc)
        rl_m["avg_cost"] = float(np.mean(rl_costs))
        rl_m["std_cost"] = float(np.std(rl_costs))
        rl_m["avg_time_s"] = float(np.mean(rl_times))
        rl_m["equivalence"] = check_equivalence(qc, best_rl_qc)
        results["rl"] = rl_m
        if verbose:
            eq_line = format_equivalence_result(rl_m["equivalence"])
            print(f"  RL cost:        {rl_m['cost']:.4f}  "
                  f"(avg={rl_m['avg_cost']:.4f} +/- {rl_m['std_cost']:.4f})  "
                  f"time={rl_m['avg_time_s']:.2f}s  equiv={eq_line}")
    else:
        results["rl"] = None
        if verbose:
            print("  RL: skipped (no trained RL agent)")

    # ── 5. GA + Simplify (expand -> simplify) ───────────────────
    ga_refined_qc = simplify(identity_based_refinement(best_ga_qc, fitness_fn=fitness))
    ga_s_m = circuit_metrics(ga_refined_qc)
    ga_s_m["equivalence"] = check_equivalence(qc, ga_refined_qc)
    results["ga_simplify"] = ga_s_m
    if verbose:
        eq_line = format_equivalence_result(ga_s_m["equivalence"])
        print(f"  GA+Simplify:    {ga_s_m['cost']:.4f}  depth={ga_s_m['depth']}  "
              f"delay={ga_s_m['delay']}  gates={ga_s_m['total_gates']}  equiv={eq_line}")

    # ── 6. RL + Simplify (expand -> simplify) ───────────────────
    if results["rl"] is not None:
        rl_base_qc = best_rl_qc
        rl_refined_qc = simplify(identity_based_refinement(rl_base_qc, fitness_fn=fitness))
        rl_s_m = circuit_metrics(rl_refined_qc)
        rl_s_m["equivalence"] = check_equivalence(qc, rl_refined_qc)
        results["rl_simplify"] = rl_s_m
        if verbose:
            eq_line = format_equivalence_result(rl_s_m["equivalence"])
            print(f"  RL+Simplify:    {rl_s_m['cost']:.4f}  depth={rl_s_m['depth']}  "
                  f"delay={rl_s_m['delay']}  gates={rl_s_m['total_gates']}  equiv={eq_line}")
    else:
        results["rl_simplify"] = None

    return results


def benchmark_suite(
    circuits: List[dict],
    rl_agent: Optional[RLAgent] = None,
    ga_generations: int = config.GA_GENERATIONS,
    ga_pop_size: int = config.GA_POPULATION_SIZE,
    runs: int = config.BENCHMARK_RUNS,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run ``benchmark_single`` on every circuit in the list.

    Parameters
    ----------
    circuits : list of {name, circuit} dicts
    """
    all_results = []
    for entry in circuits:
        res = benchmark_single(
            qc=entry["circuit"],
            name=entry["name"],
            rl_agent=rl_agent,
            ga_generations=ga_generations,
            ga_pop_size=ga_pop_size,
            runs=runs,
            verbose=verbose,
        )
        all_results.append(res)
    return all_results


def print_summary_table(results: List[Dict]):
    """Print final comparison table required by the project brief."""
    header = (
        f"{'Circuit':<18} | {'Method':<12} | {'Fidelity':>8} | {'Depth':>5} | "
        f"{'Delay':>5} | {'Gates':>5} | {'Eq':>4}"
    )
    print("\n" + "=" * len(header))
    print("  FINAL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        ordered = [
            ("ga", "GA"),
            ("rl", "RL"),
            ("ga_simplify", "GA+Simplify"),
            ("rl_simplify", "RL+Simplify"),
            ("qiskit", "Qiskit"),
        ]

        first_line = True
        for key, label in ordered:
            entry = r.get(key)
            if entry is None:
                continue
            eq = entry.get("equivalence")
            eq_flag = "PASS" if eq and eq.get("is_equivalent", False) else "FAIL"
            circuit_name = r["name"] if first_line else ""
            print(
                f"{circuit_name:<18} | {label:<12} | {entry['fidelity']:>8.4f} | "
                f"{entry['depth']:>5} | {entry['delay']:>5} | "
                f"{entry['total_gates']:>5} | {eq_flag:>4}"
            )
            first_line = False
        print("-" * len(header))

    print("=" * len(header))
