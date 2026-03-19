"""
RL Agent wrapper around Stable-Baselines3 PPO / DQN.

Provides train() and optimize() helpers that work with the
CircuitOptEnv Gymnasium environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

import config
from circuit_optimizer.environment import CircuitOptEnv
from circuit_optimizer.cost import circuit_cost
from circuit_optimizer.representation import copy_circuit


# ───────────────────────────────────────────────────────────────────
#  Training callback – records per-episode stats
# ───────────────────────────────────────────────────────────────────

class _EpisodeLogger(BaseCallback):
    """Logs per-episode improvement percentage."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_improvements: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "improvement_pct" in info:
                self.episode_improvements.append(info["improvement_pct"])
        return True


# ───────────────────────────────────────────────────────────────────
#  RLAgent
# ───────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Thin wrapper that trains a PPO (or DQN) policy for circuit
    optimisation and provides an *optimize(circuit)* method.
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        circuit_generator: Optional[Callable[[], QuantumCircuit]] = None,
        learning_rate: float = config.RL_LEARNING_RATE,
        gamma: float = config.RL_GAMMA,
        total_timesteps: int = config.RL_TOTAL_TIMESTEPS,
        n_steps: int = config.RL_N_STEPS,
        batch_size: int = config.RL_BATCH_SIZE,
        n_epochs: int = config.RL_N_EPOCHS,
        verbose: int = 1,
    ):
        self.algorithm = algorithm.upper()
        self.circuit_generator = circuit_generator
        self.lr = learning_rate
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.env: Optional[CircuitOptEnv] = None
        self.model = None
        self.logger_cb: Optional[_EpisodeLogger] = None

    # ── Training ───────────────────────────────────────────────

    def train(
        self,
        circuit_generator: Optional[Callable[[], QuantumCircuit]] = None,
    ) -> dict:
        """
        Train the RL policy.

        Returns dict with training statistics.
        """
        gen = circuit_generator or self.circuit_generator
        if gen is None:
            raise ValueError("Provide a circuit_generator.")

        self.env = CircuitOptEnv(circuit_generator=gen)
        self.logger_cb = _EpisodeLogger(verbose=self.verbose)

        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.lr,
                gamma=self.gamma,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                verbose=self.verbose,
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.lr,
                gamma=self.gamma,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.logger_cb,
        )

        return {
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "episodes_logged": len(self.logger_cb.episode_improvements),
            "avg_improvement_pct": (
                float(np.mean(self.logger_cb.episode_improvements))
                if self.logger_cb.episode_improvements else 0.0
            ),
        }

    # ── Inference (optimise a single circuit) ──────────────────

    def optimize(
        self,
        qc: QuantumCircuit,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        deterministic: bool = True,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> QuantumCircuit:
        """
        Apply the learned policy greedily to *qc* and return the
        best circuit found during the rollout.
        """
        if self.model is None:
            raise RuntimeError("Agent has not been trained yet.")

        env = CircuitOptEnv(initial_circuit=qc, max_steps=max_steps)
        obs, _ = env.reset()

        for _ in range(max_steps):
            if stop_check is not None and stop_check():
                break
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

        return env.get_best_circuit()

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str = "models/rl_agent"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str = "models/rl_agent"):
        AlgClass = PPO if self.algorithm == "PPO" else DQN
        self.model = AlgClass.load(path)

    # ── Greedy local search (key for hybrid) ──────────────────

    def refine(
        self,
        qc: QuantumCircuit,
        max_steps: int = 30,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> QuantumCircuit:
        """
        Greedy local search: at each step, try ALL actions and pick
        the one that yields the lowest cost.  Much stronger than
        policy-only rollout.  This is used by the Hybrid optimiser
        on elite individuals.
        """
        from circuit_optimizer.actions import apply_action, num_actions
        from circuit_optimizer.cost import circuit_cost
        current = copy_circuit(qc)
        current_cost = circuit_cost(current)
        best = copy_circuit(current)
        best_cost = current_cost
        n_act = num_actions()

        for _ in range(max_steps):
            if stop_check is not None and stop_check():
                break
            improved = False
            best_action_qc = None
            best_action_cost = current_cost

            for a in range(n_act):
                candidate = apply_action(current, a)
                if candidate is not None:
                    c = circuit_cost(candidate)
                    if c < best_action_cost:
                        best_action_cost = c
                        best_action_qc = candidate
                        improved = True

            if improved and best_action_qc is not None:
                current = best_action_qc
                current_cost = best_action_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = copy_circuit(current)
            else:
                break  # no improving action found

        return best

    # ── Utility for GA integration ─────────────────────────────

    def suggest_action(self, qc: QuantumCircuit) -> int:
        """
        Ask the policy for the best action given the current circuit
        (used by the hybrid GA for RL-guided mutation).
        """
        if self.model is None:
            raise RuntimeError("Agent has not been trained yet.")
        from circuit_optimizer.representation import circuit_features
        obs = circuit_features(qc)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
