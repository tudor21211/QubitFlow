"""
Custom Gymnasium environment for quantum-circuit optimisation.

Observation : feature vector  (see representation.circuit_features)
Action      : index into the rewrite-rule list
Reward      : cost_before - cost_after  (positive = improvement)
Termination : max steps, stagnation, or no applicable action
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit

import config
from circuit_optimizer.actions import apply_action, num_actions, get_rules
from circuit_optimizer.cost import circuit_cost, reward as compute_reward
from circuit_optimizer.representation import circuit_features, copy_circuit


class CircuitOptEnv(gym.Env):
    """Gymnasium environment that wraps a quantum circuit for RL."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        circuit_generator=None,
        initial_circuit: Optional[QuantumCircuit] = None,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        stagnation_limit: int = config.STAGNATION_LIMIT,
    ):
        """
        Parameters
        ----------
        circuit_generator : callable() -> QuantumCircuit
            If provided, a new circuit is sampled at every reset().
        initial_circuit : QuantumCircuit
            Fixed circuit (used if circuit_generator is None).
        """
        super().__init__()

        self._circuit_gen = circuit_generator
        self._fixed_circuit = initial_circuit

        self.n_actions = num_actions()
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.STATE_DIM,), dtype=np.float32,
        )

        self.max_steps = max_steps
        self.stagnation_limit = stagnation_limit

        # Will be set in reset()
        self.current_circuit: Optional[QuantumCircuit] = None
        self.initial_cost: float = 0.0
        self.current_cost: float = 0.0
        self.best_cost: float = 0.0
        self.best_circuit: Optional[QuantumCircuit] = None
        self.step_count: int = 0
        self.steps_since_improvement: int = 0

    # ────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._circuit_gen is not None:
            self.current_circuit = self._circuit_gen()
        elif self._fixed_circuit is not None:
            self.current_circuit = copy_circuit(self._fixed_circuit)
        else:
            raise ValueError("No circuit source configured.")

        self.initial_cost = circuit_cost(self.current_circuit)
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost
        self.best_circuit = copy_circuit(self.current_circuit)
        self.step_count = 0
        self.steps_since_improvement = 0

        obs = circuit_features(self.current_circuit)
        return obs, self._info()

    # ────────────────────────────────────────────────────────────

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Returns (obs, reward, terminated, truncated, info).
        """
        self.step_count += 1

        # Try to apply the selected rewrite rule
        new_qc = apply_action(self.current_circuit, int(action))

        # If the chosen action is not applicable, try others as fallback
        if new_qc is None:
            fallback_order = list(range(self.n_actions))
            fallback_order.remove(int(action))
            for alt in fallback_order:
                new_qc = apply_action(self.current_circuit, alt)
                if new_qc is not None:
                    break

        if new_qc is None:
            # No action applicable at all → penalty, no change
            obs = circuit_features(self.current_circuit)
            rew = -0.05
            self.steps_since_improvement += 1
        else:
            new_cost = circuit_cost(new_qc)
            rew = compute_reward(self.current_cost, new_cost)

            # Bonus reward for reducing CX count specifically
            old_cx = self.current_circuit.count_ops().get("cx", 0)
            new_cx = new_qc.count_ops().get("cx", 0)
            if new_cx < old_cx:
                rew += 0.3 * (old_cx - new_cx)

            self.current_circuit = new_qc
            self.current_cost = new_cost

            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_circuit = copy_circuit(new_qc)
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

            obs = circuit_features(self.current_circuit)

        terminated = False
        truncated = False

        # Stop conditions
        if self.step_count >= self.max_steps:
            truncated = True
        if self.steps_since_improvement >= self.stagnation_limit:
            truncated = True

        return obs, rew, terminated, truncated, self._info()

    # ────────────────────────────────────────────────────────────

    def _info(self) -> dict:
        return {
            "step": self.step_count,
            "current_cost": self.current_cost,
            "best_cost": self.best_cost,
            "initial_cost": self.initial_cost,
            "improvement_pct": (
                100.0 * (self.initial_cost - self.best_cost) / self.initial_cost
                if self.initial_cost > 0 else 0.0
            ),
        }

    def render(self, mode="human"):
        if self.current_circuit is not None:
            print(self.current_circuit.draw(output="text"))

    def get_best_circuit(self) -> QuantumCircuit:
        """Return the lowest-cost circuit found during the episode."""
        return copy_circuit(self.best_circuit)
