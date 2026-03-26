"""Runner helpers for the Streamlit visualization app."""

import inspect
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit

import config
from circuit_optimizer.circuits.generators import random_redundant_circuit
from circuit_optimizer.cost import circuit_cost, circuit_metrics
from circuit_optimizer.equivalence import check_equivalence
from circuit_optimizer.genetic_algorithm import GeneticAlgorithm
from circuit_optimizer.hybrid_optimizer import HybridOptimizer
from circuit_optimizer.representation import copy_circuit
from circuit_optimizer.rl_agent import RLAgent


class OptimizationCancelled(Exception):
    """Raised when an optimization run is cancelled by the caller."""


def _adaptive_ga_patience(total_generations: int) -> int:
    """Return an adaptive early-stop patience bounded by config limits."""
    lower = int(getattr(config, "GA_EARLY_STOP_PATIENCE_MIN", 20))
    upper = int(getattr(config, "GA_EARLY_STOP_PATIENCE_MAX", 30))
    if lower > upper:
        lower, upper = upper, lower

    scaled = max(1, total_generations // 5)
    return max(lower, min(upper, scaled))


class OptimizationResult:
    """Container for one generation + optimization run."""

    def __init__(
        self,
        original_circuit: QuantumCircuit,
        optimized_circuit: QuantumCircuit,
        original_metrics: Dict,
        optimized_metrics: Dict,
        method_used: str,
        model_message: str,
        equivalence_result: Optional[Dict] = None,
        ga_circuit: Optional[QuantumCircuit] = None,
        ga_metrics: Optional[Dict] = None,
        ga_equivalence_result: Optional[Dict] = None,
        hybrid_circuit: Optional[QuantumCircuit] = None,
        hybrid_metrics: Optional[Dict] = None,
        hybrid_equivalence_result: Optional[Dict] = None,
        run_events: Optional[List[Dict[str, Any]]] = None,
        convergence_points: Optional[List[Dict[str, Any]]] = None,
        attempt_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.original_circuit = original_circuit
        self.optimized_circuit = optimized_circuit
        self.original_metrics = original_metrics
        self.optimized_metrics = optimized_metrics
        self.method_used = method_used
        self.model_message = model_message
        self.equivalence_result = equivalence_result
        self.ga_circuit = ga_circuit
        self.ga_metrics = ga_metrics
        self.ga_equivalence_result = ga_equivalence_result
        self.hybrid_circuit = hybrid_circuit
        self.hybrid_metrics = hybrid_metrics
        self.hybrid_equivalence_result = hybrid_equivalence_result
        self.run_events = list(run_events) if run_events is not None else []
        self.convergence_points = list(convergence_points) if convergence_points is not None else []
        self.attempt_summaries = list(attempt_summaries) if attempt_summaries is not None else []


def _emit_progress(
    events: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    event_type: str,
    message: str,
    **payload: Any,
) -> None:
    """Store a progress event and optionally stream it to the UI callback."""
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event_type": event_type,
        "message": message,
    }
    event.update(payload)
    events.append(event)

    if progress_callback is not None:
        try:
            progress_callback(event)
        except Exception:
            # Progress emission should never break optimization flow.
            pass


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
    cancel_check: Optional[Callable[[], bool]] = None,
    generation_callback: Optional[Callable[[dict], None]] = None,
    early_stop_patience: Optional[int] = None,
) -> tuple[QuantumCircuit, List[dict]]:
    """Run the GA optimizer and return the best circuit found."""
    ga_kwargs: Dict[str, Any] = {
        "circuit_fn": lambda: copy_circuit(base_circuit),
        "generations": ga_generations,
        "pop_size": ga_pop_size,
        "verbose": False,
    }
    init_params = inspect.signature(GeneticAlgorithm.__init__).parameters
    if early_stop_patience is not None and "early_stop_patience" in init_params:
        ga_kwargs["early_stop_patience"] = int(early_stop_patience)

    ga = GeneticAlgorithm(
        **ga_kwargs,
    )
    run_params = inspect.signature(ga.run).parameters
    if "stop_check" in run_params and "generation_callback" in run_params:
        best_qc, history = ga.run(
            stop_check=cancel_check,
            generation_callback=generation_callback,
        )
    elif "stop_check" in run_params:
        best_qc, history = ga.run(stop_check=cancel_check)
    else:
        best_qc, history = ga.run()
    return best_qc, history


def _run_hybrid(
    base_circuit: QuantumCircuit,
    rl_agent: RLAgent,
    ga_generations: int,
    ga_pop_size: int,
    cancel_check: Optional[Callable[[], bool]] = None,
    generation_callback: Optional[Callable[[dict], None]] = None,
    early_stop_patience: Optional[int] = None,
) -> tuple[QuantumCircuit, List[dict]]:
    """Run the hybrid optimizer and return the best circuit found."""
    hybrid_kwargs: Dict[str, Any] = {
        "circuit_fn": lambda: copy_circuit(base_circuit),
        "rl_agent": rl_agent,
        "generations": ga_generations,
        "pop_size": ga_pop_size,
        "verbose": False,
    }
    init_params = inspect.signature(HybridOptimizer.__init__).parameters
    if early_stop_patience is not None and "early_stop_patience" in init_params:
        hybrid_kwargs["early_stop_patience"] = int(early_stop_patience)

    hybrid = HybridOptimizer(
        **hybrid_kwargs,
    )
    run_params = inspect.signature(hybrid.run).parameters
    if "stop_check" in run_params and "generation_callback" in run_params:
        best_qc, history = hybrid.run(
            stop_check=cancel_check,
            generation_callback=generation_callback,
        )
    elif "stop_check" in run_params:
        best_qc, history = hybrid.run(stop_check=cancel_check)
    else:
        best_qc, history = hybrid.run()

    if cancel_check is not None and cancel_check():
        return best_qc, history

    # Optional polish step with the learned policy.
    if "stop_check" in inspect.signature(rl_agent.optimize).parameters:
        polished = rl_agent.optimize(best_qc, stop_check=cancel_check)
    else:
        polished = rl_agent.optimize(best_qc)
    if circuit_cost(polished) < circuit_cost(best_qc):
        return polished, history
    return best_qc, history


def _raise_if_cancelled(
    cancel_check: Optional[Callable[[], bool]],
    events: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    phase: str = "cancel",
) -> None:
    """Raise OptimizationCancelled when cancellation is requested."""
    if cancel_check is not None and cancel_check():
        _emit_progress(
            events,
            progress_callback,
            event_type="run-cancelled",
            message="Optimization cancelled.",
            attempt=attempt,
            max_attempts=max_attempts,
            phase=phase,
        )
        raise OptimizationCancelled("Optimization cancelled by request.")


def generate_and_optimize(
    n_qubits: int,
    depth: int,
    seed: Optional[int],
    ga_generations: int,
    ga_pop_size: int,
    model_path: str,
    max_attempts: int = 3,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> OptimizationResult:
    """Generate random circuits and optimize until gate reduction is found or attempts end."""
    rl_agent, model_message = _try_load_agent(model_path)
    run_events: List[Dict[str, Any]] = []
    convergence_points: List[Dict[str, Any]] = []
    attempt_summaries: List[Dict[str, Any]] = []
    ga_patience = _adaptive_ga_patience(ga_generations)

    _emit_progress(
        run_events,
        progress_callback,
        event_type="run-start",
        message="Starting Generate and Optimize run.",
        max_attempts=max_attempts,
    )
    _raise_if_cancelled(cancel_check, run_events, progress_callback, phase="start")

    best_result: Optional[OptimizationResult] = None
    best_gate_delta = 10**9

    for attempt in range(max_attempts):
        attempt_no = attempt + 1
        trial_seed = None if seed is None else seed + attempt
        _raise_if_cancelled(
            cancel_check,
            run_events,
            progress_callback,
            attempt=attempt_no,
            max_attempts=max_attempts,
            phase="attempt-start",
        )
        _emit_progress(
            run_events,
            progress_callback,
            event_type="attempt-start",
            message=f"Attempt {attempt_no}/{max_attempts}: generating random circuit.",
            attempt=attempt_no,
            max_attempts=max_attempts,
            generation=0,
            total_generations=ga_generations,
            phase="generate",
        )

        original = random_redundant_circuit(n_qubits=n_qubits, depth=depth, seed=trial_seed)
        original_metrics = circuit_metrics(original)
        _raise_if_cancelled(
            cancel_check,
            run_events,
            progress_callback,
            attempt=attempt_no,
            max_attempts=max_attempts,
            phase="generate",
        )

        _emit_progress(
            run_events,
            progress_callback,
            event_type="circuit-generated",
            message=(
                f"Attempt {attempt_no}/{max_attempts}: generated circuit with "
                f"{original_metrics['total_gates']} gates and depth {original_metrics['depth']}."
            ),
            attempt=attempt_no,
            max_attempts=max_attempts,
            phase="generate",
            original_metrics=original_metrics,
        )

        if rl_agent is not None and rl_agent.model is not None:
            _emit_progress(
                run_events,
                progress_callback,
                event_type="phase-start",
                message=f"Attempt {attempt_no}/{max_attempts}: running GA-only baseline.",
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="ga",
                generation=0,
                total_generations=ga_generations,
            )
            ga_progress_streamed = False

            def _on_ga_generation(point: Dict[str, Any]) -> None:
                nonlocal ga_progress_streamed
                ga_progress_streamed = True
                convergence_points.append(
                    {
                        "attempt": attempt_no,
                        "branch": "GA",
                        "generation": int(point["generation"]) + 1,
                        "best_cost": float(point["best_cost"]),
                        "avg_cost": float(-point["avg_fitness"]),
                    }
                )
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="generation",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts} [GA] gen "
                        f"{int(point['generation']) + 1}/{ga_generations}: "
                        f"best={float(point['best_cost']):.4f}, "
                        f"avg={float(-point['avg_fitness']):.4f}"
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="ga",
                    branch="GA",
                    generation=int(point["generation"]) + 1,
                    total_generations=ga_generations,
                    best_cost=float(point["best_cost"]),
                    avg_cost=float(-point["avg_fitness"]),
                )

            ga_optimized = _run_ga(
                base_circuit=original,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
                cancel_check=cancel_check,
                generation_callback=_on_ga_generation,
                early_stop_patience=ga_patience,
            )
            ga_optimized, ga_history = ga_optimized
            _raise_if_cancelled(
                cancel_check,
                run_events,
                progress_callback,
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="ga",
            )
            if not ga_progress_streamed:
                for point in ga_history:
                    _on_ga_generation(point)

            if len(ga_history) < ga_generations:
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="phase-end",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts}: GA early-stopped at "
                        f"generation {len(ga_history)}/{ga_generations} "
                        f"(patience={ga_patience})."
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="ga",
                    generation=len(ga_history),
                    total_generations=ga_generations,
                )

            _emit_progress(
                run_events,
                progress_callback,
                event_type="phase-start",
                message=f"Attempt {attempt_no}/{max_attempts}: running Hybrid (GA+RL) with refinement.",
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="hybrid",
                generation=0,
                total_generations=ga_generations,
            )
            hybrid_progress_streamed = False

            def _on_hybrid_generation(point: Dict[str, Any]) -> None:
                nonlocal hybrid_progress_streamed
                hybrid_progress_streamed = True
                convergence_points.append(
                    {
                        "attempt": attempt_no,
                        "branch": "Hybrid",
                        "generation": int(point["generation"]) + 1,
                        "best_cost": float(point["best_cost"]),
                        "avg_cost": float(-point["avg_fitness"]),
                    }
                )
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="generation",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts} [Hybrid] gen "
                        f"{int(point['generation']) + 1}/{ga_generations}: "
                        f"best={float(point['best_cost']):.4f}, "
                        f"avg={float(-point['avg_fitness']):.4f}"
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="hybrid",
                    branch="Hybrid",
                    generation=int(point["generation"]) + 1,
                    total_generations=ga_generations,
                    best_cost=float(point["best_cost"]),
                    avg_cost=float(-point["avg_fitness"]),
                )

            hybrid_optimized = _run_hybrid(
                base_circuit=original,
                rl_agent=rl_agent,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
                cancel_check=cancel_check,
                generation_callback=_on_hybrid_generation,
                early_stop_patience=ga_patience,
            )
            hybrid_optimized, hybrid_history = hybrid_optimized
            _raise_if_cancelled(
                cancel_check,
                run_events,
                progress_callback,
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="hybrid",
            )
            if not hybrid_progress_streamed:
                for point in hybrid_history:
                    _on_hybrid_generation(point)

            if len(hybrid_history) < ga_generations:
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="phase-end",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts}: Hybrid GA early-stopped at "
                        f"generation {len(hybrid_history)}/{ga_generations} "
                        f"(patience={ga_patience})."
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="hybrid",
                    generation=len(hybrid_history),
                    total_generations=ga_generations,
                )

            _emit_progress(
                run_events,
                progress_callback,
                event_type="phase-end",
                message=f"Attempt {attempt_no}/{max_attempts}: hybrid GA loop complete, applying policy polish.",
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="hybrid",
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
            _emit_progress(
                run_events,
                progress_callback,
                event_type="phase-start",
                message=(
                    f"Attempt {attempt_no}/{max_attempts}: RL model unavailable, running GA-only optimization."
                ),
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="ga",
                generation=0,
                total_generations=ga_generations,
            )
            ga_only_progress_streamed = False

            def _on_ga_only_generation(point: Dict[str, Any]) -> None:
                nonlocal ga_only_progress_streamed
                ga_only_progress_streamed = True
                convergence_points.append(
                    {
                        "attempt": attempt_no,
                        "branch": "GA",
                        "generation": int(point["generation"]) + 1,
                        "best_cost": float(point["best_cost"]),
                        "avg_cost": float(-point["avg_fitness"]),
                    }
                )
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="generation",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts} [GA] gen "
                        f"{int(point['generation']) + 1}/{ga_generations}: "
                        f"best={float(point['best_cost']):.4f}, "
                        f"avg={float(-point['avg_fitness']):.4f}"
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="ga",
                    branch="GA",
                    generation=int(point["generation"]) + 1,
                    total_generations=ga_generations,
                    best_cost=float(point["best_cost"]),
                    avg_cost=float(-point["avg_fitness"]),
                )

            optimized = _run_ga(
                base_circuit=original,
                ga_generations=ga_generations,
                ga_pop_size=ga_pop_size,
                cancel_check=cancel_check,
                generation_callback=_on_ga_only_generation,
                early_stop_patience=ga_patience,
            )
            optimized, ga_history = optimized
            _raise_if_cancelled(
                cancel_check,
                run_events,
                progress_callback,
                attempt=attempt_no,
                max_attempts=max_attempts,
                phase="ga",
            )
            if not ga_only_progress_streamed:
                for point in ga_history:
                    _on_ga_only_generation(point)

            if len(ga_history) < ga_generations:
                _emit_progress(
                    run_events,
                    progress_callback,
                    event_type="phase-end",
                    message=(
                        f"Attempt {attempt_no}/{max_attempts}: GA early-stopped at "
                        f"generation {len(ga_history)}/{ga_generations} "
                        f"(patience={ga_patience})."
                    ),
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    phase="ga",
                    generation=len(ga_history),
                    total_generations=ga_generations,
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
        attempt_summary = {
            "attempt": attempt_no,
            "seed": trial_seed,
            "method_used": method_used,
            "original_gates": original_metrics["total_gates"],
            "optimized_gates": optimized_metrics["total_gates"],
            "gate_delta": gate_delta,
            "ga_cost": ga_metrics["cost"] if ga_metrics is not None else None,
            "hybrid_cost": hybrid_metrics["cost"] if hybrid_metrics is not None else None,
            "selected_cost": optimized_metrics["cost"],
        }
        attempt_summaries.append(attempt_summary)

        _emit_progress(
            run_events,
            progress_callback,
            event_type="attempt-result",
            message=(
                f"Attempt {attempt_no}/{max_attempts}: method={method_used}, "
                f"gate delta={gate_delta}, selected cost={optimized_metrics['cost']:.4f}"
            ),
            max_attempts=max_attempts,
            phase="summary",
            **attempt_summary,
        )

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
            run_events=list(run_events),
            convergence_points=list(convergence_points),
            attempt_summaries=list(attempt_summaries),
        )

        if gate_delta < best_gate_delta:
            best_gate_delta = gate_delta
            best_result = current

        if gate_delta < 0:
            _emit_progress(
                run_events,
                progress_callback,
                event_type="run-end",
                message=f"Early success on attempt {attempt_no}/{max_attempts}: gate count reduced.",
                attempt=attempt_no,
                max_attempts=max_attempts,
            )
            current.run_events = list(run_events)
            current.convergence_points = list(convergence_points)
            current.attempt_summaries = list(attempt_summaries)
            return current

    _emit_progress(
        run_events,
        progress_callback,
        event_type="run-end",
        message="Run finished after max attempts. Returning best result found.",
        attempt=max_attempts,
        max_attempts=max_attempts,
    )
    if best_result is not None:
        best_result.run_events = list(run_events)
        best_result.convergence_points = list(convergence_points)
        best_result.attempt_summaries = list(attempt_summaries)

    return best_result
