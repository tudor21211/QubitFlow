#!/usr/bin/env python
"""
main.py  –  Entry point for the Quantum Circuit Optimiser.

Usage
-----
    python main.py                   # full pipeline (train + benchmark)
    python main.py --mode train      # train RL agent only
    python main.py --mode benchmark  # benchmark only (needs saved model)
    python main.py --mode demo       # quick demo on one circuit
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import subprocess

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import config
from circuit_optimizer.circuits.generators import (
    all_test_circuits,
    make_generator,
    random_redundant_circuit,
)
from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.rl_agent import RLAgent
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm
from circuit_optimizer.hybrid_optimizer import HybridOptimizer
from circuit_optimizer.baseline import transpile_baseline
from circuit_optimizer.benchmark import (
    benchmark_suite,
    print_summary_table,
)
from circuit_optimizer.visualization import plot_all, plot_ga_convergence
from circuit_optimizer.equivalence import check_equivalence, format_equivalence_result


def parse_args():
    p = argparse.ArgumentParser(description="Quantum Circuit Optimiser (RL + GA)")
    p.add_argument("--mode", choices=["full", "train", "benchmark", "demo", "visualize"],
                    default="full", help="Execution mode")
    p.add_argument("--rl-timesteps", type=int, default=config.RL_TOTAL_TIMESTEPS,
                    help="Total RL training timesteps")
    p.add_argument("--ga-gens", type=int, default=config.GA_GENERATIONS,
                    help="GA generations")
    p.add_argument("--ga-pop", type=int, default=config.GA_POPULATION_SIZE,
                    help="GA population size")
    p.add_argument("--runs", type=int, default=config.BENCHMARK_RUNS,
                    help="Benchmark repetitions per method")
    p.add_argument(
        "--ga-selection-mode",
        type=str,
        choices=["weighted", "pareto"],
        default=config.GA_SELECTION_MODE,
        help="GA parent selection mode",
    )
    p.add_argument(
        "--selection-ablation",
        action="store_true",
        help="In benchmark mode, evaluate both weighted and pareto and print ablation rows",
    )
    p.add_argument("--model-path", type=str, default="models/rl_agent",
                    help="Path for saving / loading RL model")
    p.add_argument("--qubits", type=int, default=4,
                    help="Number of qubits for training circuits")
    p.add_argument("--depth", type=int, default=20,
                    help="Depth of random training circuits")
    p.add_argument("--seed", type=int, default=None,
                    help="Optional seed for reproducible random circuits")
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────
#  Modes
# ───────────────────────────────────────────────────────────────────

def run_demo(args):
    """Quick demo: optimise one circuit with all methods."""
    print("\n" + "=" * 60)
    print("  DEMO: Optimising a single random circuit")
    print("=" * 60)
    print(f"GA selection mode: {config.GA_SELECTION_MODE}")

    n_q = max(args.qubits, 5)
    d = max(args.depth, 50)
    qc = random_redundant_circuit(n_qubits=n_q, depth=d, seed=7)
    orig_m = circuit_metrics(qc)
    print(f"\nOriginal ({n_q}q, depth-target {d}):")
    print(f"  cost={orig_m['cost']:.4f}  depth={orig_m['depth']}  "
          f"cx={orig_m['cx_count']}  total_gates={orig_m['total_gates']}")

    # Baseline
    base_qc = transpile_baseline(qc)
    base_m = circuit_metrics(base_qc)
    print(f"\nBaseline:  cost={base_m['cost']:.4f}  depth={base_m['depth']}  "
          f"cx={base_m['cx_count']}  total_gates={base_m['total_gates']}")

    n_gens = min(args.ga_gens, 30)
    n_pop = min(args.ga_pop, 25)

    # GA only
    print(f"\nRunning GA only ({n_gens} gens, {n_pop} pop)...")
    ga = GeneticAlgorithm(
        circuit_fn=lambda: copy_circuit(qc),
        generations=n_gens,
        pop_size=n_pop,
    )
    ga_qc, ga_hist = ga.run()
    ga_m = circuit_metrics(ga_qc)
    print(f"GA:        cost={ga_m['cost']:.4f}  depth={ga_m['depth']}  "
          f"cx={ga_m['cx_count']}  total_gates={ga_m['total_gates']}")

    # Train RL agent
    rl_steps = min(args.rl_timesteps, 20_000)
    print(f"\nTraining RL agent ({rl_steps} steps)...")
    gen = make_generator(n_qubits=n_q, depth=d)
    agent = RLAgent(
        algorithm="PPO",
        circuit_generator=gen,
        total_timesteps=rl_steps,
        verbose=0,
    )
    agent.train()

    rl_qc = agent.optimize(qc)
    rl_m = circuit_metrics(rl_qc)
    print(f"RL only:   cost={rl_m['cost']:.4f}  depth={rl_m['depth']}  "
          f"cx={rl_m['cx_count']}  total_gates={rl_m['total_gates']}")

    # Hybrid GA+RL (RL-guided mutation + RL local search on elites)
    print(f"\nRunning Hybrid GA+RL ({n_gens} gens, {n_pop} pop)...")
    hybrid = HybridOptimizer(
        circuit_fn=lambda: copy_circuit(qc),
        rl_agent=agent,
        generations=n_gens,
        pop_size=n_pop,
        rl_refine_top_k=5,
        rl_refine_steps=20,
        verbose=True,
    )
    h_qc, h_hist = hybrid.run()
    h_m = circuit_metrics(h_qc)
    print(f"Hybrid:    cost={h_m['cost']:.4f}  depth={h_m['depth']}  "
          f"cx={h_m['cx_count']}  total_gates={h_m['total_gates']}")

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Method':<15} {'Cost':>8} {'Depth':>7} {'CX':>5} {'Gates':>7}")
    print("-" * 60)
    for label, m in [("Original", orig_m), ("Baseline", base_m),
                     ("GA only", ga_m), ("RL only", rl_m),
                     ("Hybrid GA+RL", h_m)]:
        print(f"  {label:<15} {m['cost']:>8.4f} {m['depth']:>7} "
              f"{m['cx_count']:>5} {m['total_gates']:>7}")
    print("=" * 60)

    ga_imp = 100.0 * (orig_m['cost'] - ga_m['cost']) / orig_m['cost'] if orig_m['cost'] > 0 else 0
    h_imp = 100.0 * (orig_m['cost'] - h_m['cost']) / orig_m['cost'] if orig_m['cost'] > 0 else 0
    print(f"  GA improvement vs Original:     {ga_imp:.1f}%")
    print(f"  Hybrid improvement vs Original: {h_imp:.1f}%")
    if ga_m['cost'] > 0:
        h_vs_ga = 100.0 * (ga_m['cost'] - h_m['cost']) / ga_m['cost']
        print(f"  Hybrid improvement vs GA:       {h_vs_ga:.1f}%")

    # ── Equivalence verification ───────────────────────────────
    print("\n" + "=" * 60)
    print("  EQUIVALENCE VERIFICATION  (optimised vs original)")
    print("=" * 60)
    equivalence_pairs = [
        ("Baseline",    base_qc),
        ("GA only",     ga_qc),
        ("RL only",     rl_qc),
        ("Hybrid GA+RL", h_qc),
    ]
    all_pass = True
    for method_label, opt_qc in equivalence_pairs:
        result = check_equivalence(qc, opt_qc)
        line = format_equivalence_result(result, label=method_label)
        print(f"  {line}")
        if not result["is_equivalent"]:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("  All optimised circuits are functionally equivalent to the original.")
    else:
        print("  WARNING: one or more optimised circuits are NOT equivalent to the original!")

    # Plot GA convergence
    plot_ga_convergence(ga_hist, title="Demo \u2013 GA Convergence")


def run_train(args):
    """Train the RL agent and save."""
    print("\n=== Training RL Agent ===\n")
    gen = make_generator(n_qubits=args.qubits, depth=args.depth)
    agent = RLAgent(
        algorithm="PPO",
        circuit_generator=gen,
        total_timesteps=args.rl_timesteps,
    )
    stats = agent.train()
    agent.save(args.model_path)
    print(f"\nModel saved to: {args.model_path}")
    print(f"Stats: {stats}")
    return agent


def run_benchmark(args, agent: RLAgent | None = None):
    """Benchmark on the full test suite."""
    if agent is None:
        print("Loading RL model...")
        agent = RLAgent(algorithm="PPO")
        try:
            agent.load(args.model_path)
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running benchmark without Hybrid method.")
            agent = None

    circuits = all_test_circuits()
    print(f"\nBenchmarking {len(circuits)} circuits...\n")

    results = benchmark_suite(
        circuits=circuits,
        rl_agent=agent,
        ga_generations=args.ga_gens,
        ga_pop_size=args.ga_pop,
        runs=args.runs,
        selection_modes=["weighted", "pareto"] if args.selection_ablation else None,
    )

    print_summary_table(results)

    # Plots
    plot_all(results)

    return results


def run_full(args):
    """Full pipeline: train → benchmark → visualise."""
    agent = run_train(args)
    results = run_benchmark(args, agent=agent)
    return results


def run_visualize(args):
    """Launch the Streamlit UI for interactive circuit visualization."""
    app_path = os.path.join(
        os.path.dirname(__file__),
        "circuit_optimizer",
        "interactive_viz",
        "app.py",
    )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        app_path,
        "--",
        "--qubits",
        str(args.qubits),
        "--depth",
        str(args.depth),
        "--ga-gens",
        str(args.ga_gens),
        "--ga-pop",
        str(args.ga_pop),
        "--model-path",
        str(args.model_path),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print("Launching visualization app...")
    subprocess.run(cmd, check=False)


# ───────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config.GA_SELECTION_MODE = args.ga_selection_mode.lower()
    print(f"Using GA selection mode: {config.GA_SELECTION_MODE}")

    mode_fn = {
        "demo": run_demo,
        "train": run_train,
        "benchmark": run_benchmark,
        "full": run_full,
        "visualize": run_visualize,
    }

    t0 = time.time()
    mode_fn[args.mode](args)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
