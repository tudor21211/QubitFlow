"""Streamlit app for automatic circuit generation and optimization visualization."""

from __future__ import annotations

import argparse
import base64
import html
import inspect
import mimetypes
import os
import sys
import tempfile
import threading
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

import config

# Streamlit Cloud may execute this file with the app folder as the import base.
# Ensure repository root is in sys.path so `circuit_optimizer` is importable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from circuit_optimizer.interactive_viz.plotting import save_artifacts
from circuit_optimizer.interactive_viz.runner import (
    OptimizationCancelled,
    OptimizationResult,
    generate_and_optimize,
)


_RUN_CONTROL_LOCK = threading.Lock()
_RUN_CONTROL: Dict[str, Any] = {
    "active_run_id": None,
    "cancel_flags": {},
}


def _mark_existing_run_cancelled() -> Optional[str]:
    """Cancel currently active run, if any, and return its ID."""
    with _RUN_CONTROL_LOCK:
        active_run_id = _RUN_CONTROL.get("active_run_id")
        if active_run_id is None:
            return None
        _RUN_CONTROL["cancel_flags"][active_run_id] = True
        return active_run_id


def _start_run() -> str:
    """Create and register a new run ID."""
    run_id = uuid.uuid4().hex
    with _RUN_CONTROL_LOCK:
        _RUN_CONTROL["active_run_id"] = run_id
        _RUN_CONTROL["cancel_flags"][run_id] = False
    return run_id


def _is_cancelled(run_id: str) -> bool:
    """Return whether cancellation has been requested for run_id."""
    with _RUN_CONTROL_LOCK:
        return bool(_RUN_CONTROL["cancel_flags"].get(run_id, False))


def _finish_run(run_id: str) -> None:
    """Release run tracking for a completed/cancelled run."""
    with _RUN_CONTROL_LOCK:
        _RUN_CONTROL["cancel_flags"].pop(run_id, None)
        if _RUN_CONTROL.get("active_run_id") == run_id:
            _RUN_CONTROL["active_run_id"] = None


def _image_to_data_url(file_path: str) -> str:
    """Convert a local image file to an inline data URL."""
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "image/png"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _render_click_modal_styles() -> None:
    """Inject CSS for click-to-open centered modal image viewer."""
    st.markdown(
        """
        <style>
        .modal-trigger {
            position: relative;
            text-align: center;
        }
        .modal-trigger img {
            width: 100%;
            max-height: 230px;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            cursor: zoom-in;
            background: #ffffff;
        }
        .modal-trigger .cap {
            font-size: 0.9rem;
            opacity: 0.85;
            margin-top: 0.35rem;
            text-align: center;
        }
        .img-modal {
            position: fixed;
            inset: 0;
            opacity: 0;
            pointer-events: none;
            background: rgba(10, 12, 18, 0.24);
            backdrop-filter: blur(7px);
            -webkit-backdrop-filter: blur(7px);
            z-index: 9999;
            transition: opacity 0.15s ease;
        }
        .img-modal:target {
            opacity: 1;
            pointer-events: auto;
        }
        .img-modal .modal-inner {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3vh 2vw;
            box-sizing: border-box;
        }
        .img-modal img {
            max-width: min(94vw, 1750px);
            max-height: 84vh;
            border-radius: 10px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.42);
            background: #ffffff;
        }
        .img-modal .modal-caption {
            margin-top: 0.5rem;
            text-align: center;
            color: #ffffff;
            font-size: 0.95rem;
            opacity: 0.92;
        }
        .img-modal .close-btn {
            position: fixed;
            top: 14px;
            right: 18px;
            color: #ffffff;
            text-decoration: none;
            font-size: 1.6rem;
            background: rgba(0, 0, 0, 0.35);
            border-radius: 8px;
            padding: 0.12rem 0.5rem;
            line-height: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_click_modal_image(file_path: str, caption: str, modal_id: str) -> None:
    """Render image with click-to-open centered modal and blurred background."""
    data_url = _image_to_data_url(file_path)
    safe_caption = html.escape(caption)
    safe_modal_id = html.escape(modal_id)
    st.markdown(
        f"""
        <div class="modal-trigger">
            <a href="#{safe_modal_id}">
                <img src="{data_url}" />
            </a>
            <div class="cap">{safe_caption}</div>
        </div>
        <div id="{safe_modal_id}" class="img-modal">
            <a href="#" class="close-btn" aria-label="Close">x</a>
            <a href="#" class="modal-inner">
                <div>
                    <img src="{data_url}" />
                    <div class="modal-caption">{safe_caption}</div>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_plain_image_with_caption(file_path: str, caption: str) -> None:
    """Render a normal non-modal image with caption."""
    data_url = _image_to_data_url(file_path)
    safe_caption = html.escape(caption)
    st.markdown(
        f"""
        <div class="modal-trigger">
            <img src="{data_url}" style="cursor: default;" />
            <div class="cap">{safe_caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _secret_value(*keys: str) -> Optional[str]:
    """Return the first matching non-empty secret value from common key styles."""
    try:
        for key in keys:
            if key in st.secrets:
                value = st.secrets[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        model_section = st.secrets.get("model")
        if isinstance(model_section, dict):
            for key in keys:
                if key in model_section:
                    value = model_section[key]
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    except StreamlitSecretNotFoundError:
        return None
    return None


@st.cache_data(show_spinner=False)
def _download_cloud_model(model_url: str, auth_header: Optional[str], auth_token: Optional[str]) -> str:
    """Download model zip to temp storage and return SB3 base model path."""
    model_dir = os.path.join(tempfile.gettempdir(), "qubitflow_models")
    os.makedirs(model_dir, exist_ok=True)
    model_base = os.path.join(model_dir, "rl_agent")
    model_zip = model_base + ".zip"

    req = urllib.request.Request(model_url)
    if auth_token:
        req.add_header(auth_header or "Authorization", f"Bearer {auth_token}")

    with urllib.request.urlopen(req, timeout=120) as response, open(model_zip, "wb") as out:
        out.write(response.read())

    return model_base


def _resolve_default_model_path(local_default: str) -> tuple[str, Optional[str]]:
    """Resolve model path, preferring cloud-secret model URL if available."""
    model_url = _secret_value("RL_MODEL_URL", "rl_model_url", "url")
    if not model_url:
        return local_default, None

    auth_header = _secret_value("RL_MODEL_AUTH_HEADER", "rl_model_auth_header", "auth_header")
    auth_token = _secret_value("RL_MODEL_AUTH_TOKEN", "rl_model_auth_token", "auth_token")

    try:
        cloud_path = _download_cloud_model(model_url, auth_header, auth_token)
        return cloud_path, "Loaded RL model from cloud secret URL."
    except Exception as exc:
        return local_default, f"Cloud model download failed: {exc}. Using local model path setting."


def _launch_defaults() -> argparse.Namespace:
    """Parse optional defaults passed after `streamlit run ... --` from main.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--ga-gens", type=int, default=30)
    parser.add_argument("--ga-pop", type=int, default=25)
    parser.add_argument("--model-path", type=str, default="models/rl_agent")
    parser.add_argument("--seed", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args


def _build_metrics_table(result: OptimizationResult) -> pd.DataFrame:
    """Create a small metrics table for UI display."""
    before = result.original_metrics
    after = result.optimized_metrics
    rows = [
        {
            "Metric": "Total gates",
            "Before": before["total_gates"],
            "After": after["total_gates"],
            "Improvement %": round(100.0 * (before["total_gates"] - after["total_gates"]) / max(before["total_gates"], 1), 2),
        },
        {
            "Metric": "Depth",
            "Before": before["depth"],
            "After": after["depth"],
            "Improvement %": round(100.0 * (before["depth"] - after["depth"]) / max(before["depth"], 1), 2),
        },
        {
            "Metric": "CX gates",
            "Before": before["cx_count"],
            "After": after["cx_count"],
            "Improvement %": round(100.0 * (before["cx_count"] - after["cx_count"]) / max(before["cx_count"], 1), 2),
        },
        {
            "Metric": "Cost",
            "Before": round(before["cost"], 4),
            "After": round(after["cost"], 4),
            "Improvement %": round(100.0 * (before["cost"] - after["cost"]) / max(before["cost"], 1e-9), 2),
        },
    ]
    return pd.DataFrame(rows)


def _build_method_comparison_table(result: OptimizationResult) -> pd.DataFrame:
    """Compare GA-only and Hybrid metrics when both are available."""
    ga_metrics = getattr(result, "ga_metrics", None)
    hybrid_metrics = getattr(result, "hybrid_metrics", None)
    if ga_metrics is None or hybrid_metrics is None:
        return pd.DataFrame()

    ga = ga_metrics
    hy = hybrid_metrics
    return pd.DataFrame(
        [
            {"Metric": "Total gates", "GA": ga["total_gates"], "Hybrid": hy["total_gates"]},
            {"Metric": "Depth", "GA": ga["depth"], "Hybrid": hy["depth"]},
            {"Metric": "CX gates", "GA": ga["cx_count"], "Hybrid": hy["cx_count"]},
            {"Metric": "Cost", "GA": round(ga["cost"], 4), "Hybrid": round(hy["cost"], 4)},
        ]
    )


def _init_run_state() -> None:
    """Initialize session state used for live run telemetry."""
    if "live_events" not in st.session_state:
        st.session_state.live_events = []
    if "live_chart_points" not in st.session_state:
        st.session_state.live_chart_points = []
    if "live_attempt" not in st.session_state:
        st.session_state.live_attempt = 0
    if "live_max_attempts" not in st.session_state:
        st.session_state.live_max_attempts = 1
    if "live_generation" not in st.session_state:
        st.session_state.live_generation = 0
    if "live_total_generations" not in st.session_state:
        st.session_state.live_total_generations = 1
    if "live_phase" not in st.session_state:
        st.session_state.live_phase = "idle"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_artifacts" not in st.session_state:
        st.session_state.last_artifacts = None
    if "show_activity_details" not in st.session_state:
        st.session_state.show_activity_details = False
    if "reload_cancel_notice" not in st.session_state:
        st.session_state.reload_cancel_notice = None


def _reset_live_run_state() -> None:
    """Reset live telemetry buffers before each new optimization run."""
    st.session_state.live_events = []
    st.session_state.live_chart_points = []
    st.session_state.live_attempt = 0
    st.session_state.live_max_attempts = 1
    st.session_state.live_generation = 0
    st.session_state.live_total_generations = 1
    st.session_state.live_phase = "starting"


def _event_line(event: Dict[str, Any]) -> str:
    """Format one progress event as a compact log line."""
    ts = event.get("timestamp", "--:--:--")
    return f"[{ts}] {event.get('message', '')}"


def _render_live_activity(
    attempt_placeholder,
    generation_placeholder,
    chart_placeholder,
    log_placeholder,
    status_placeholder,
    running: bool,
    show_details: bool,
) -> None:
    """Render the activity panel from current session-state telemetry."""
    attempt = int(st.session_state.live_attempt)
    max_attempts = max(int(st.session_state.live_max_attempts), 1)
    generation = int(st.session_state.live_generation)
    total_generations = max(int(st.session_state.live_total_generations), 1)
    phase = str(st.session_state.live_phase)

    attempt_progress = 0.0
    if max_attempts > 0:
        attempt_progress = min(max(attempt / max_attempts, 0.0), 1.0)
    attempt_placeholder.progress(
        attempt_progress,
        text=f"Attempts: {attempt}/{max_attempts}",
    )

    generation_progress = 0.0
    if total_generations > 0:
        generation_progress = min(max(generation / total_generations, 0.0), 1.0)
    generation_placeholder.progress(
        generation_progress,
        text=f"Generation Progress ({phase}): {generation}/{total_generations}",
    )

    chart_points = st.session_state.live_chart_points
    if show_details:
        if chart_points:
            chart_df = pd.DataFrame(chart_points)
            chart_df["step"] = chart_df.index + 1
            chart_placeholder.line_chart(
                chart_df.set_index("step")[ ["best_cost", "avg_cost"] ],
                height=220,
                use_container_width=True,
            )
        else:
            chart_placeholder.info("Convergence chart will populate once GA generations start.")
    else:
        chart_placeholder.info("Convergence history is hidden. Click 'Show Logs and Convergence' to view it.")

    lines = [_event_line(event) for event in st.session_state.live_events]
    if not lines:
        lines = ["Waiting to start optimization..."]

    if show_details:
        visible_lines = lines[-600:]
        rendered_lines = "".join(
            f"<div style='margin: 0 0 0.15rem 0;'>{html.escape(line)}</div>" for line in visible_lines
        )
        log_placeholder.markdown(
            f"""
            <div style="
                max-height: 290px;
                overflow-y: auto;
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 0.75rem;
                color: #e2e8f0;
                font-size: 0.80rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
                line-height: 1.35;
            ">
                {rendered_lines}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        log_placeholder.info("Execution log is hidden. Click 'Show Logs and Convergence' to display it.")

    if running:
        status_placeholder.info("Optimization in progress. Live events update below.")
    else:
        status_placeholder.success("Optimization finished. Log and convergence history are preserved.")


def main() -> None:
    defaults = _launch_defaults()
    resolved_model_path, cloud_model_message = _resolve_default_model_path(defaults.model_path)

    st.set_page_config(page_title="Circuit Optimizer Visualizer", layout="wide")
    _render_click_modal_styles()
    st.title("Circuit Optimizer Visualizer")
    st.write(
        "Automatically generate a random quantum circuit, optimize it, and compare before/after artifacts."
    )

    st.sidebar.header("Circuit Setup")
    n_qubits = st.sidebar.slider("Qubits", min_value=2, max_value=10, value=defaults.qubits)
    depth = st.sidebar.slider("Random depth", min_value=8, max_value=200, value=defaults.depth)
    seed_enabled = st.sidebar.checkbox("Use fixed seed", value=defaults.seed is not None)
    seed: Optional[int] = None
    if seed_enabled:
        seed_default = defaults.seed if defaults.seed is not None else 7
        seed = st.sidebar.number_input("Seed", min_value=0, value=seed_default, step=1)
        st.sidebar.caption(
            f"Special seeds: {config.SEED_IDENTITY_REFINEMENT} (identity refinement), "
            f"{config.SEED_TOFFOLI_DEMO} (Toffoli-heavy circuit)."
        )

    st.sidebar.header("Optimizer Setup")
    ga_generations = st.sidebar.slider("GA generations", min_value=5, max_value=100, value=defaults.ga_gens)
    ga_pop_size = st.sidebar.slider("GA population", min_value=8, max_value=80, value=defaults.ga_pop)
    model_path = st.sidebar.text_input("RL model path", value=resolved_model_path)
    if cloud_model_message:
        st.sidebar.caption(cloud_model_message)

    _init_run_state()

    cancelled_run_id = _mark_existing_run_cancelled()
    if cancelled_run_id is not None:
        st.session_state.reload_cancel_notice = cancelled_run_id

    if st.session_state.reload_cancel_notice:
        st.warning(
            "Detected a page reload while optimization was running. "
            "The previous run was cancelled to keep the app responsive."
        )
        st.session_state.reload_cancel_notice = None

    st.subheader("Live Optimization Activity")
    st.caption(
        "This panel streams the code path in real time: circuit generation, GA/hybrid phases, and per-generation convergence."
    )
    toggle_label = "Hide Logs and Convergence" if st.session_state.show_activity_details else "Show Logs and Convergence"
    if st.button(toggle_label, key="toggle-live-details"):
        st.session_state.show_activity_details = not st.session_state.show_activity_details

    pipeline_cols = st.columns(3)
    pipeline_cols[0].markdown(
        "**Generate**  \n"
        "`random_redundant_circuit(...)` creates a new candidate with redundant structure to optimize."
    )
    pipeline_cols[1].markdown(
        "**Optimize**  \n"
        "`GeneticAlgorithm.run()` evolves rewrite sequences; if RL model is loaded, `HybridOptimizer.run()` is also executed."
    )
    pipeline_cols[2].markdown(
        "**Validate**  \n"
        "`check_equivalence(...)` verifies that optimized and original circuits implement the same transformation."
    )

    attempt_progress_placeholder = st.empty()
    generation_progress_placeholder = st.empty()
    convergence_chart_placeholder = st.empty()
    live_log_placeholder = st.empty()
    live_status_placeholder = st.empty()

    _render_live_activity(
        attempt_placeholder=attempt_progress_placeholder,
        generation_placeholder=generation_progress_placeholder,
        chart_placeholder=convergence_chart_placeholder,
        log_placeholder=live_log_placeholder,
        status_placeholder=live_status_placeholder,
        running=False,
        show_details=st.session_state.show_activity_details,
    )

    run_clicked = st.button("Generate and Optimize", type="primary")

    if run_clicked:
        _reset_live_run_state()
        run_id = _start_run()

        def _on_progress(event: Dict[str, Any]) -> None:
            st.session_state.live_events.append(event)
            st.session_state.live_phase = event.get("phase", st.session_state.live_phase)
            st.session_state.live_attempt = int(event.get("attempt", st.session_state.live_attempt))
            st.session_state.live_max_attempts = int(
                event.get("max_attempts", st.session_state.live_max_attempts)
            )
            st.session_state.live_generation = int(event.get("generation", st.session_state.live_generation))
            st.session_state.live_total_generations = int(
                event.get("total_generations", st.session_state.live_total_generations)
            )

            if event.get("event_type") == "generation":
                st.session_state.live_chart_points.append(
                    {
                        "best_cost": float(event.get("best_cost", 0.0)),
                        "avg_cost": float(event.get("avg_cost", 0.0)),
                    }
                )

            _render_live_activity(
                attempt_placeholder=attempt_progress_placeholder,
                generation_placeholder=generation_progress_placeholder,
                chart_placeholder=convergence_chart_placeholder,
                log_placeholder=live_log_placeholder,
                status_placeholder=live_status_placeholder,
                running=True,
                show_details=st.session_state.show_activity_details,
            )

        optimize_kwargs = {
            "n_qubits": n_qubits,
            "depth": depth,
            "seed": int(seed) if seed is not None else None,
            "ga_generations": ga_generations,
            "ga_pop_size": ga_pop_size,
            "model_path": model_path,
        }

        if "progress_callback" in inspect.signature(generate_and_optimize).parameters:
            optimize_kwargs["progress_callback"] = _on_progress
        else:
            st.warning(
                "Live progress streaming is not available with the current runner version. "
                "Run-level summary and artifacts will still be shown."
            )

        if "cancel_check" in inspect.signature(generate_and_optimize).parameters:
            optimize_kwargs["cancel_check"] = lambda rid=run_id: _is_cancelled(rid)

        try:
            result = generate_and_optimize(**optimize_kwargs)
            artifacts = save_artifacts(result)
        except OptimizationCancelled:
            st.session_state.live_phase = "cancelled"
            st.session_state.live_events.append(
                {
                    "timestamp": "--:--:--",
                    "event_type": "run-cancelled",
                    "message": "Optimization was cancelled (page reload or another run started).",
                    "phase": "cancel",
                }
            )
            _render_live_activity(
                attempt_placeholder=attempt_progress_placeholder,
                generation_placeholder=generation_progress_placeholder,
                chart_placeholder=convergence_chart_placeholder,
                log_placeholder=live_log_placeholder,
                status_placeholder=live_status_placeholder,
                running=False,
                show_details=st.session_state.show_activity_details,
            )
            st.warning("Optimization cancelled.")
            _finish_run(run_id)
            return
        except Exception:
            _finish_run(run_id)
            raise

        _finish_run(run_id)

        st.session_state.last_result = result
        st.session_state.last_artifacts = artifacts

        if getattr(result, "run_events", None):
            st.session_state.live_events = list(result.run_events)
        if getattr(result, "convergence_points", None):
            st.session_state.live_chart_points = [
                {
                    "best_cost": float(point["best_cost"]),
                    "avg_cost": float(point["avg_cost"]),
                }
                for point in result.convergence_points
            ]

        _render_live_activity(
            attempt_placeholder=attempt_progress_placeholder,
            generation_placeholder=generation_progress_placeholder,
            chart_placeholder=convergence_chart_placeholder,
            log_placeholder=live_log_placeholder,
            status_placeholder=live_status_placeholder,
            running=False,
            show_details=st.session_state.show_activity_details,
        )

    result = st.session_state.last_result
    artifacts = st.session_state.last_artifacts

    if result is None or artifacts is None:
        st.info("Pick parameters on the left, then click Generate and Optimize.")
        return

    before_gates = result.original_metrics["total_gates"]
    after_gates = result.optimized_metrics["total_gates"]
    reduced = before_gates - after_gates

    st.subheader("Run Summary")
    st.caption(result.model_message)
    cols = st.columns(3)
    cols[0].metric("Method used", result.method_used)
    cols[1].metric("Before gates", before_gates)
    cols[2].metric("After gates", after_gates, delta=f"{reduced} gates")

    if reduced > 0:
        st.success(f"Optimization reduced the circuit by {reduced} gates.")
    elif reduced == 0:
        before_cost = float(result.original_metrics.get("cost", 0.0))
        after_cost = float(result.optimized_metrics.get("cost", 0.0))
        if after_cost < before_cost:
            st.info(
                "Gate count is unchanged, but overall cost improved "
                f"({before_cost:.4f} -> {after_cost:.4f})."
            )
        else:
            st.warning("No gate-count reduction this run. Try a higher depth or more GA generations.")
    else:
        st.warning("Gate count increased in this run. Cost may still be improved; see the metrics table.")

    st.subheader("Metrics")
    st.dataframe(_build_metrics_table(result), use_container_width=True)

    if getattr(result, "attempt_summaries", None):
        with st.expander("Run Timeline"):
            attempt_df = pd.DataFrame(result.attempt_summaries)
            st.dataframe(attempt_df, use_container_width=True)

    ga_metrics = getattr(result, "ga_metrics", None)
    hybrid_metrics = getattr(result, "hybrid_metrics", None)
    if ga_metrics is not None and hybrid_metrics is not None:
        st.subheader("GA vs Hybrid")
        ga_cost = ga_metrics["cost"]
        hybrid_cost = hybrid_metrics["cost"]
        delta_cost = ga_cost - hybrid_cost

        cmp_cols = st.columns(3)
        cmp_cols[0].metric("GA cost", f"{ga_cost:.4f}")
        cmp_cols[1].metric("Hybrid cost", f"{hybrid_cost:.4f}")
        cmp_cols[2].metric("Hybrid gain vs GA", f"{delta_cost:.4f}")

        if delta_cost > 0:
            st.success("Hybrid achieved lower cost than GA-only on this run.")
        elif delta_cost == 0:
            st.info("GA-only and Hybrid produced the same cost on this run.")
        else:
            st.warning("GA-only achieved lower cost than Hybrid on this run.")

        st.dataframe(_build_method_comparison_table(result), use_container_width=True)

    img_cols = st.columns(3)
    with img_cols[0]:
        _render_click_modal_image(
            artifacts["before"],
            os.path.basename(artifacts["before"]),
            modal_id="before-modal",
        )
    with img_cols[1]:
        _render_click_modal_image(
            artifacts["after"],
            os.path.basename(artifacts["after"]),
            modal_id="after-modal",
        )
    with img_cols[2]:
        _render_plain_image_with_caption(artifacts["metrics"], os.path.basename(artifacts["metrics"]))

    if artifacts.get("ga_only") and artifacts.get("hybrid"):
        st.subheader("GA-only vs GA+RL Circuit Images")
        compare_cols = st.columns(2)
        with compare_cols[0]:
            _render_click_modal_image(
                artifacts["ga_only"],
                os.path.basename(artifacts["ga_only"]),
                modal_id="ga-only-modal",
            )
        with compare_cols[1]:
            _render_click_modal_image(
                artifacts["hybrid"],
                os.path.basename(artifacts["hybrid"]),
                modal_id="hybrid-modal",
            )

    # ── Equivalence Verification ───────────────────────────────
    st.subheader("Equivalence Verification")
    eq = result.equivalence_result
    if eq is None:
        st.warning("Equivalence check not available for this run.")
    else:
        if eq.get("error"):
            st.error(f"Equivalence check failed: {eq['error']}")
        elif eq["is_equivalent"]:
            st.success(
                "**EQUIVALENT** — The optimized circuit implements the same unitary "
                "transformation as the original (up to global phase)."
            )
        else:
            st.error(
                "**NOT EQUIVALENT** — The optimized circuit does NOT match the original. "
                "One or more rewrite rules may have introduced a logical error."
            )

        # Key metrics row
        eq_cols = st.columns(3)
        eq_cols[0].metric("Method", eq["method"].capitalize())
        if eq["method"] == "unitary":
            eq_cols[1].metric("Max element diff", f"{eq['max_abs_diff']:.2e}")
            eq_cols[2].metric("Verdict", "PASS" if eq["is_equivalent"] else "FAIL")
        else:
            eq_cols[1].metric("Min fidelity", f"{eq['min_fidelity']:.8f}")
            eq_cols[2].metric("Max fidelity error", f"{eq['max_fidelity_error']:.2e}")

        # Equivalence plot
        if artifacts.get("equivalence"):
            st.image(artifacts["equivalence"], use_container_width=True)

        with st.expander("What does this check?"):
            if eq["method"] == "unitary":
                st.markdown(
                    "The full **unitary matrix** $U$ (size $2^n \\times 2^n$) is computed for "
                    "both circuits using Qiskit's `Operator` simulator. The matrices are compared "
                    "element-wise after removing the global phase (which is physically unobservable). "
                    "A max element difference below $10^{-6}$ means the circuits are functionally "
                    "identical."
                )
            else:
                st.markdown(
                    "The circuits are simulated on the **$|0\\cdots0\\rangle$ computational basis "
                    "state** and several **Haar-random input states**. For each input $|\\psi\\rangle$, "
                    "the state fidelity $|\\langle\\phi_1|\\phi_2\\rangle|^2$ between the two output "
                    "states is computed. A fidelity of 1 for all inputs means the circuits are "
                    "functionally equivalent."
                )

    # ── ASCII Circuit Views ────────────────────────────────────
    st.subheader("ASCII Circuit Views")
    text_cols = st.columns(2)
    text_cols[0].code(result.original_circuit.draw(output="text"), language="text")
    text_cols[1].code(result.optimized_circuit.draw(output="text"), language="text")


if __name__ == "__main__":
    main()
