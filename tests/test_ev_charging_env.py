"""Automated tests for the EV charging Gymnasium environment."""

import numpy as np
import pytest

from ev_charging_env import EVChargingEnv


def test_reset_initializes_valid_observation_and_state():
    env = EVChargingEnv()
    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert observation.tolist() == [0, 4, info["price_level"]]
    assert info["battery_level"] == 4
    assert info["target_battery"] == 8
    assert info["price_level"] in (0, 1, 2)


def test_reset_is_reproducible_when_seed_is_reused():
    env = EVChargingEnv()
    first_observation, _ = env.reset(seed=42)
    second_observation, _ = env.reset(seed=42)

    assert np.array_equal(first_observation, second_observation)


def test_slow_charge_adds_one_level_and_uses_current_price():
    env = EVChargingEnv()
    env.reset(seed=1)
    env.price_level = 0  # Set low price so the cost is deterministic.

    observation, reward, terminated, truncated, info = env.step(EVChargingEnv.SLOW_CHARGE)

    assert observation.tolist()[1] == 5
    assert info["energy_added_kwh"] == pytest.approx(6.0)
    assert info["electricity_cost"] == pytest.approx(0.60)
    assert reward == pytest.approx(-0.60)
    assert terminated is False
    assert truncated is False


def test_fast_charge_adds_two_levels_and_applies_wear_penalty():
    env = EVChargingEnv()
    env.reset(seed=2)
    env.price_level = 0

    observation, reward, _, _, info = env.step(EVChargingEnv.FAST_CHARGE)

    assert observation.tolist()[1] == 6
    assert info["energy_added_kwh"] == pytest.approx(12.0)
    assert info["electricity_cost"] == pytest.approx(1.20)
    assert info["wear_penalty"] == pytest.approx(0.50)
    assert reward == pytest.approx(-1.70)


def test_battery_never_exceeds_capacity():
    env = EVChargingEnv()
    env.reset(seed=3)
    env.battery_level = 9
    env.price_level = 0

    observation, _, _, _, info = env.step(EVChargingEnv.FAST_CHARGE)

    assert observation.tolist()[1] == 10
    assert info["levels_added"] == 1
    assert info["energy_added_kwh"] == pytest.approx(6.0)


def test_departure_terminates_and_applies_shortfall_penalty():
    env = EVChargingEnv()
    env.reset(seed=4)
    env.hour = 13
    env.battery_level = 4
    env.price_level = 0

    observation, reward, terminated, truncated, info = env.step(EVChargingEnv.IDLE)

    assert observation.tolist()[0] == 14
    assert terminated is True
    assert truncated is False
    assert info["shortfall_penalty"] == pytest.approx(20.0)
    assert reward == pytest.approx(-20.0)


def test_step_after_episode_ends_requires_reset():
    env = EVChargingEnv()
    env.reset(seed=5)
    env.hour = 13
    env.step(EVChargingEnv.IDLE)

    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(EVChargingEnv.IDLE)


def test_price_transition_rows_are_valid_probabilities():
    env = EVChargingEnv()

    assert np.allclose(env.base_transition_matrix.sum(axis=1), 1.0)
    assert np.allclose(env.peak_transition_matrix.sum(axis=1), 1.0)
    assert np.all(env.base_transition_matrix >= 0.0)
    assert np.all(env.peak_transition_matrix >= 0.0)
