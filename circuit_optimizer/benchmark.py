"""
Benchmarking harness.

Compares four approaches on a suite of test circuits:
  1. Original (no optimisation)
  2. Qiskit transpiler baseline
  3. GA only
  4. Hybrid GA + RL
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit

import config
from circuit_optimizer.baseline import transpile_baseline
from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm
from circuit_optimizer.hybrid_optimizer import HybridOptimizer
from circuit_optimizer.rl_agent import RLAgent
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.equivalence import check_equivalence, format_equivalence_result


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

    # ── 2. Qiskit baseline ─────────────────────────────────────
    t0 = time.time()
    base_qc = transpile_baseline(qc, optimization_level=3)
    base_time = time.time() - t0
    base_m = circuit_metrics(base_qc)
    base_m["time_s"] = base_time
    base_m["equivalence"] = check_equivalence(qc, base_qc)
    results["baseline"] = base_m
    if verbose:
        eq_line = format_equivalence_result(base_m["equivalence"])
        print(f"  Baseline cost:  {base_m['cost']:.4f}  "
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

    # ── 4. Hybrid GA + RL ──────────────────────────────────────
    if rl_agent is not None and rl_agent.model is not None:
        hybrid_costs = []
        hybrid_times = []
        best_hybrid_qc = None
        best_hybrid_cost = np.inf

        for r in range(runs):
            t0 = time.time()
            hybrid = HybridOptimizer(
                circuit_fn=circuit_fn,
                rl_agent=rl_agent,
                generations=ga_generations,
                pop_size=ga_pop_size,
                verbose=False,
            )
            h_qc, _ = hybrid.run()
            # Also do RL polish
            polished = rl_agent.optimize(h_qc)
            if circuit_cost(polished) < circuit_cost(h_qc):
                h_qc = polished
            elapsed = time.time() - t0
            c = circuit_cost(h_qc)
            hybrid_costs.append(c)
            hybrid_times.append(elapsed)
            if c < best_hybrid_cost:
                best_hybrid_cost = c
                best_hybrid_qc = h_qc

        hybrid_m = circuit_metrics(best_hybrid_qc)
        hybrid_m["avg_cost"] = float(np.mean(hybrid_costs))
        hybrid_m["std_cost"] = float(np.std(hybrid_costs))
        hybrid_m["avg_time_s"] = float(np.mean(hybrid_times))
        hybrid_m["equivalence"] = check_equivalence(qc, best_hybrid_qc)
        results["hybrid"] = hybrid_m
        if verbose:
            eq_line = format_equivalence_result(hybrid_m["equivalence"])
            print(f"  Hybrid cost:    {hybrid_m['cost']:.4f}  "
                  f"(avg={hybrid_m['avg_cost']:.4f} +/- {hybrid_m['std_cost']:.4f})  "
                  f"time={hybrid_m['avg_time_s']:.2f}s  equiv={eq_line}")
    else:
        results["hybrid"] = None
        if verbose:
            print("  Hybrid: skipped (no trained RL agent)")

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
    """Print a compact text comparison table."""
    header = (f"{'Circuit':<20} | {'Original':>10} | {'Baseline':>10} | "
              f"{'GA':>10} | {'Hybrid':>10} | "
              f"{'Baseline Eq':>11} | {'GA Eq':>5} | {'Hybrid Eq':>9}")
    print("\n" + "=" * len(header))
    print("  COST COMPARISON  (lower is better)  +  EQUIVALENCE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        orig = f"{r['original']['cost']:.2f}"
        base = f"{r['baseline']['cost']:.2f}"
        ga = f"{r['ga']['cost']:.2f}"
        hyb = f"{r['hybrid']['cost']:.2f}" if r.get("hybrid") else "N/A"

        def _eq_flag(method_key: str) -> str:
            entry = r.get(method_key)
            if not entry:
                return "N/A"
            eq = entry.get("equivalence")
            if eq is None:
                return "N/A"
            return "PASS" if eq["is_equivalent"] else "FAIL"

        base_eq = _eq_flag("baseline")
        ga_eq = _eq_flag("ga")
        hyb_eq = _eq_flag("hybrid")

        print(f"{r['name']:<20} | {orig:>10} | {base:>10} | {ga:>10} | {hyb:>10} | "
              f"{base_eq:>11} | {ga_eq:>5} | {hyb_eq:>9}")

    print("=" * len(header))
