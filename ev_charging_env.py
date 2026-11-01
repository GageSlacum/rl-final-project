"""Custom Gymnasium environment for cost-aware EV charging.

Observation: [hour, battery_level, price_level]
Actions: 0 = idle, 1 = slow charge (+1 battery level), 2 = fast charge (+2 levels)

The environment uses a 14-hour episode. Each battery level represents 10% of a
60 kWh battery (6 kWh). Rewards are negative dollar-equivalent costs.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EVChargingEnv(gym.Env):
    """A discrete EV charging environment with uncertain electricity prices."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    IDLE = 0
    SLOW_CHARGE = 1
    FAST_CHARGE = 2

    ACTION_NAMES = {
        IDLE: "Idle",
        SLOW_CHARGE: "Slow charge",
        FAST_CHARGE: "Fast charge",
    }

    PRICE_NAMES = {0: "Low", 1: "Medium", 2: "High"}

    def __init__(
        self,
        render_mode: str | None = None,
        time_horizon: int = 14,
        initial_battery: int = 4,
        target_battery: int = 8,
        battery_capacity: int = 10,
        energy_per_level_kwh: float = 6.0,
        price_levels: tuple[float, float, float] = (0.10, 0.20, 0.35),
        fast_charge_wear_penalty: float = 0.50,
        shortfall_penalty_per_level: float = 5.0,
    ) -> None:
        super().__init__()

        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'.")
        if time_horizon < 1:
            raise ValueError("time_horizon must be at least 1.")
        if not 0 <= initial_battery <= battery_capacity:
            raise ValueError("initial_battery must be within battery capacity.")
        if not 0 <= target_battery <= battery_capacity:
            raise ValueError("target_battery must be within battery capacity.")
        if battery_capacity != 10:
            raise ValueError("This milestone version uses exactly 10 battery levels.")
        if len(price_levels) != 3 or any(price < 0 for price in price_levels):
            raise ValueError("price_levels must contain three nonnegative prices.")
        if energy_per_level_kwh <= 0:
            raise ValueError("energy_per_level_kwh must be positive.")

        self.render_mode = render_mode
        self.time_horizon = time_horizon
        self.initial_battery = initial_battery
        self.target_battery = target_battery
        self.battery_capacity = battery_capacity
        self.energy_per_level_kwh = float(energy_per_level_kwh)
        self.price_levels = tuple(float(price) for price in price_levels)
        self.fast_charge_wear_penalty = float(fast_charge_wear_penalty)
        self.shortfall_penalty_per_level = float(shortfall_penalty_per_level)

        # There are 14 decision hours (0 through 13) and one terminal
        # observation at hour 14. The terminal observation is included in
        # the observation space so the final transition remains valid.
        self.observation_space = spaces.MultiDiscrete(
            np.array([self.time_horizon + 1, self.battery_capacity + 1, 3], dtype=np.int64)
        )
        self.action_space = spaces.Discrete(3)

        # Rows represent the current price (low, medium, high); columns
        # represent the next price (low, medium, high).
        self.base_transition_matrix = np.array(
            [
                [0.70, 0.25, 0.05],
                [0.20, 0.60, 0.20],
                [0.05, 0.25, 0.70],
            ],
            dtype=np.float64,
        )
        # During the final four hours, model a morning peak with a greater
        # chance of medium/high prices. This is a simple time-dependent model.
        self.peak_transition_matrix = np.array(
            [
                [0.45, 0.40, 0.15],
                [0.10, 0.55, 0.35],
                [0.03, 0.17, 0.80],
            ],
            dtype=np.float64,
        )
        self.initial_price_probabilities = np.array([0.50, 0.35, 0.15], dtype=np.float64)

        self.hour = 0
        self.battery_level = self.initial_battery
        self.price_level = 0

    def _get_obs(self) -> np.ndarray:
        """Convert the current environment state into an observation array."""
        return np.array(
            [self.hour, self.battery_level, self.price_level], dtype=np.int64
        )

    def _get_price_transition_probabilities(self, hour: int) -> np.ndarray:
        """Return the next-price probabilities for the current time of day."""
        if hour >= self.time_horizon - 4:
            return self.peak_transition_matrix[self.price_level]
        return self.base_transition_matrix[self.price_level]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new 14-hour charging episode."""
        super().reset(seed=seed)

        self.hour = 0
        self.battery_level = self.initial_battery
        self.price_level = int(
            self.np_random.choice(3, p=self.initial_price_probabilities)
        )

        observation = self._get_obs()
        info = {
            "hour": self.hour,
            "battery_level": self.battery_level,
            "price_level": self.price_level,
            "price_name": self.PRICE_NAMES[self.price_level],
            "price_per_kwh": self.price_levels[self.price_level],
            "target_battery": self.target_battery,
        }
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one charging action and advance the environment by one hour."""
        if self.hour >= self.time_horizon:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Choose 0, 1, or 2.")

        battery_before = self.battery_level
        price_level_used = self.price_level
        price_per_kwh = self.price_levels[price_level_used]

        if action == self.IDLE:
            requested_levels = 0
        elif action == self.SLOW_CHARGE:
            requested_levels = 1
        else:
            requested_levels = 2

        # Enforce the battery capacity limit.
        levels_added = min(requested_levels, self.battery_capacity - self.battery_level)
        self.battery_level += levels_added
        energy_added_kwh = levels_added * self.energy_per_level_kwh

        electricity_cost = energy_added_kwh * price_per_kwh
        wear_penalty = 0.0
        if action == self.FAST_CHARGE and levels_added > 0:
            wear_penalty = self.fast_charge_wear_penalty

        # One action represents one hour. The price used for charging is the
        # price observed before the action; the next price is then sampled.
        self.hour += 1
        terminated = self.hour >= self.time_horizon
        truncated = False

        shortfall_penalty = 0.0
        if terminated:
            battery_shortfall = max(0, self.target_battery - self.battery_level)
            shortfall_penalty = battery_shortfall * self.shortfall_penalty_per_level
        else:
            transition_probabilities = self._get_price_transition_probabilities(self.hour)
            self.price_level = int(
                self.np_random.choice(3, p=transition_probabilities)
            )

        total_cost = electricity_cost + wear_penalty + shortfall_penalty
        reward = -float(total_cost)

        info = {
            "hour": self.hour,
            "action": int(action),
            "action_name": self.ACTION_NAMES[int(action)],
            "battery_before": battery_before,
            "battery_level": self.battery_level,
            "battery_after": self.battery_level,
            "price_level_used": price_level_used,
            "price_name_used": self.PRICE_NAMES[price_level_used],
            "price_per_kwh": price_per_kwh,
            "levels_added": levels_added,
            "energy_added_kwh": energy_added_kwh,
            "electricity_cost": electricity_cost,
            "wear_penalty": wear_penalty,
            "shortfall_penalty": shortfall_penalty,
            "total_cost": total_cost,
            "target_battery": self.target_battery,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """Return or print a human-readable summary of the current state."""
        message = (
            f"Hour: {self.hour:02d}/{self.time_horizon} | "
            f"Battery: {self.battery_level * 10}% | "
            f"Price: {self.PRICE_NAMES[self.price_level]} "
            f"(${self.price_levels[self.price_level]:.2f}/kWh)"
        )
        if self.render_mode == "human":
            print(message)
            return None
        if self.render_mode == "ansi":
            return message
        return None

    def close(self) -> None:
        """No external display resources are used by this environment."""
        return None
