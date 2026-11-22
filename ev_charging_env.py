"""A simple Gymnasium environment for an electric vehicle charging problem.

The observation is [hour, battery_level, price_level].

Actions:
    0 = Do not charge
    1 = Slow charge (add up to 10% battery)
    2 = Fast charge (add up to 20% battery)

The battery has 10 levels. Each level represents 10% of a 60 kWh battery,
so one level is 6 kWh. Rewards are negative costs.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EVChargingEnv(gym.Env):
    """Simulate charging an EV while electricity prices change over time."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    # Names used when printing information about actions and prices.
    ACTION_NAMES = {
        0: "Idle",
        1: "Slow charge",
        2: "Fast charge",
    }

    PRICE_NAMES = {
        0: "Low",
        1: "Medium",
        2: "High",
    }

    # Action numbers also tell us how many 10% battery levels to request.
    IDLE = 0
    SLOW_CHARGE = 1
    FAST_CHARGE = 2

    def __init__(
        self,
        render_mode=None,
        time_horizon=14,
        initial_battery=4,
        target_battery=8,
        battery_capacity=10,
        energy_per_level_kwh=6.0,
        price_levels=(0.10, 0.20, 0.35),
        fast_charge_wear_penalty=0.50,
        shortfall_penalty_per_level=5.0,
    ):
        super().__init__()

        # Check that the settings make sense.
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'.")
        if time_horizon < 1:
            raise ValueError("time_horizon must be at least 1.")
        if battery_capacity != 10:
            raise ValueError("This version uses exactly 10 battery levels.")
        if initial_battery < 0 or initial_battery > battery_capacity:
            raise ValueError("initial_battery must be within battery capacity.")
        if target_battery < 0 or target_battery > battery_capacity:
            raise ValueError("target_battery must be within battery capacity.")
        if len(price_levels) != 3 or any(price < 0 for price in price_levels):
            raise ValueError("price_levels must contain three nonnegative prices.")
        if energy_per_level_kwh <= 0:
            raise ValueError("energy_per_level_kwh must be positive.")

        # Store the settings used by the environment.
        self.render_mode = render_mode
        self.time_horizon = time_horizon
        self.initial_battery = initial_battery
        self.target_battery = target_battery
        self.battery_capacity = battery_capacity
        self.energy_per_level_kwh = float(energy_per_level_kwh)
        self.price_levels = tuple(float(price) for price in price_levels)
        self.fast_charge_wear_penalty = float(fast_charge_wear_penalty)
        self.shortfall_penalty_per_level = float(shortfall_penalty_per_level)

        # The observation contains hour, battery level, and price level.
        # Hour can reach time_horizon because the final observation is at hour 14.
        self.observation_space = spaces.MultiDiscrete(
            [self.time_horizon + 1, self.battery_capacity + 1, 3]
        )
        self.action_space = spaces.Discrete(3)

        # Each row gives the next-price probabilities for the current price.
        # The columns represent low, medium, and high prices.
        self.base_transition_matrix = np.array(
            [
                [0.70, 0.25, 0.05],
                [0.20, 0.60, 0.20],
                [0.05, 0.25, 0.70],
            ]
        )

        # Use this matrix during the last four hours to model a morning peak.
        self.peak_transition_matrix = np.array(
            [
                [0.45, 0.40, 0.15],
                [0.10, 0.55, 0.35],
                [0.03, 0.17, 0.80],
            ]
        )

        # Starting probabilities for low, medium, and high prices.
        self.initial_price_probabilities = [0.50, 0.35, 0.15]

        # These values are set again whenever reset() is called.
        self.hour = 0
        self.battery_level = self.initial_battery
        self.price_level = 0

    def _get_obs(self):
        """Return the current state as a NumPy array."""
        return np.array(
            [self.hour, self.battery_level, self.price_level], dtype=np.int64
        )

    def _get_price_transition_probabilities(self, hour):
        """Return the next-price probabilities for the current hour."""
        if hour >= self.time_horizon - 4:
            return self.peak_transition_matrix[self.price_level]

        return self.base_transition_matrix[self.price_level]

    def reset(self, *, seed=None, options=None):
        """Start a new charging episode."""
        super().reset(seed=seed)

        self.hour = 0
        self.battery_level = self.initial_battery

        # Randomly choose the starting electricity price.
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

    def step(self, action):
        """Apply one action, calculate its cost, and move to the next hour."""
        if self.hour >= self.time_horizon:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError("Invalid action. Choose 0, 1, or 2.")

        # Save the battery and price before applying the action.
        battery_before = self.battery_level
        price_level_used = self.price_level
        price_per_kwh = self.price_levels[price_level_used]

        # The action number is also the number of battery levels requested.
        requested_levels = int(action)

        # Do not let the battery go above its maximum capacity.
        space_left = self.battery_capacity - self.battery_level
        levels_added = min(requested_levels, space_left)
        self.battery_level += levels_added

        # Calculate electricity cost and any fast-charging wear penalty.
        energy_added_kwh = levels_added * self.energy_per_level_kwh
        electricity_cost = energy_added_kwh * price_per_kwh

        wear_penalty = 0.0
        if action == self.FAST_CHARGE and levels_added > 0:
            wear_penalty = self.fast_charge_wear_penalty

        # One action represents one hour of charging.
        self.hour += 1
        terminated = self.hour >= self.time_horizon
        truncated = False

        # At departure, charge a penalty for every level below the target.
        shortfall_penalty = 0.0
        if terminated:
            battery_shortfall = max(0, self.target_battery - self.battery_level)
            shortfall_penalty = (
                battery_shortfall * self.shortfall_penalty_per_level
            )
        else:
            # Otherwise, randomly choose the next electricity price.
            price_probabilities = self._get_price_transition_probabilities(self.hour)
            self.price_level = int(
                self.np_random.choice(3, p=price_probabilities)
            )

        # The reward is the negative of all costs for this hour.
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

    def render(self):
        """Display a short description of the current state."""
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

    def close(self):
        """Close the environment. No extra cleanup is needed."""
        pass
