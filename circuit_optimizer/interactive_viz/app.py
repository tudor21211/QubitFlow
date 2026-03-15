"""Streamlit app for automatic circuit generation and optimization visualization."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from typing import Optional

import pandas as pd
import streamlit as st

# Streamlit Cloud may execute this file with the app folder as the import base.
# Ensure repository root is in sys.path so `circuit_optimizer` is importable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from circuit_optimizer.interactive_viz.plotting import save_artifacts
from circuit_optimizer.interactive_viz.runner import OptimizationResult, generate_and_optimize


def _secret_value(*keys: str) -> Optional[str]:
    """Return the first matching non-empty secret value from common key styles."""
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


def main() -> None:
    defaults = _launch_defaults()
    resolved_model_path, cloud_model_message = _resolve_default_model_path(defaults.model_path)

    st.set_page_config(page_title="Circuit Optimizer Visualizer", layout="wide")
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

    st.sidebar.header("Optimizer Setup")
    ga_generations = st.sidebar.slider("GA generations", min_value=5, max_value=100, value=defaults.ga_gens)
    ga_pop_size = st.sidebar.slider("GA population", min_value=8, max_value=80, value=defaults.ga_pop)
    model_path = st.sidebar.text_input("RL model path", value=resolved_model_path)
    if cloud_model_message:
        st.sidebar.caption(cloud_model_message)

    run_clicked = st.button("Generate and Optimize", type="primary")
    if not run_clicked:
        st.info("Pick parameters on the left, then click Generate and Optimize.")
        return

    with st.spinner("Running optimizer..."):
        result = generate_and_optimize(
            n_qubits=n_qubits,
            depth=depth,
            seed=int(seed) if seed is not None else None,
            ga_generations=ga_generations,
            ga_pop_size=ga_pop_size,
            model_path=model_path,
        )
        artifacts = save_artifacts(result)

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
        st.warning("No gate-count reduction this run. Try a higher depth or more GA generations.")
    else:
        st.warning("Gate count increased in this run. Cost may still be improved; see the metrics table.")

    st.subheader("Metrics")
    st.dataframe(_build_metrics_table(result), use_container_width=True)

    img_cols = st.columns(3)
    img_cols[0].image(artifacts["before"], caption=os.path.basename(artifacts["before"]))
    img_cols[1].image(artifacts["after"], caption=os.path.basename(artifacts["after"]))
    img_cols[2].image(artifacts["metrics"], caption=os.path.basename(artifacts["metrics"]))

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
                "✅  **EQUIVALENT** — The optimized circuit implements the same unitary "
                "transformation as the original (up to global phase)."
            )
        else:
            st.error(
                "❌  **NOT EQUIVALENT** — The optimized circuit does NOT match the original. "
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
