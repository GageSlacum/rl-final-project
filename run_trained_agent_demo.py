
"""
Demonstrate the trained Q-learning agent in one EV charging episode.

This script loads the existing Q-table. It does not retrain the agent.
"""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ev_charging_env import EVChargingEnv
from train_q_learning import QLearningAgent, TRAINING_SEED

# SETTINGS
ROOT = Path(__file__).resolve().parent

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"

Q_TABLE_PATH = OUTPUTS_DIR / "trained_q_table.npy"

# This is the first seed used in the 500-episode evaluation.
DEMO_SEED = 10000

# Fixed agent seed makes tie-breaking reproducible.
DEMO_AGENT_SEED = 2026

# Pause briefly between steps so the demonstration is easier to record.
STEP_DELAY_SECONDS = 0.4


def save_demo_csv(records):
    """Save the agent's decisions and episode costs."""

    path = OUTPUTS_DIR / "demo_episode.csv"

    with path.open("w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))

        writer.writeheader()
        writer.writerows(records)

    return path


def create_demo_figure(battery_history, price_history):
    """Create a graph showing battery and electricity price changes."""

    hours = np.arange(len(battery_history))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Battery level throughout the episode.
    axes[0].plot(hours, battery_history, marker="o", label="Battery level")
    axes[0].axhline(y=80, linestyle="--", label="Required battery (80%)")
    axes[0].set_title("Battery Level During the Charging Episode")
    axes[0].set_ylabel("Battery (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Electricity price used during each charging hour.
    price_hours = np.arange(len(price_history)) + 0.5

    axes[1].bar(price_hours, price_history, width=0.7)
    axes[1].set_title("Electricity Price During the Charging Episode")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Electricity price ($/kWh)")
    axes[1].set_xticks(hours)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle(f"Trained Q-Learning Agent Demonstration (Seed {DEMO_SEED})")
    figure.tight_layout()
    figure_path = FIGURES_DIR / "demo_episode.png"
    figure.savefig(figure_path, dpi=180)
    print(f"\nSaved demonstration graph: {figure_path}")

    return figure


def main():

    # Make sure the output folders exist.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    #Load the environment and trained Q-table.
    print("\n" + "=" * 60)
    print("       TRAINED Q-LEARNING AGENT DEMONSTRATION")
    print("=" * 60)

    if not Q_TABLE_PATH.exists():
        raise FileNotFoundError(
            "Trained Q-table not found.\n"
            "Expected location: outputs/trained_q_table.npy\n"
            "Run python train_q_learning.py first."
        )

    env = EVChargingEnv()
    agent = QLearningAgent(env, seed=DEMO_AGENT_SEED)

    loaded_q_table = np.load(Q_TABLE_PATH)
    expected_shape = (tuple(int(value) for value in env.observation_space.nvec) + (env.action_space.n,))

    if loaded_q_table.shape != expected_shape:
        env.close()
        raise ValueError(
            f"Q-table shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {loaded_q_table.shape}"
        )

    # Replace the newly initialized table with the trained table.
    agent.q_table = loaded_q_table

    print("\nTrained Q-table loaded successfully.")
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"Episode seed: {DEMO_SEED}")
    print(f"Training seed: {TRAINING_SEED}")

    #Start a new charging episode.

    observation, reset_info = env.reset(seed=DEMO_SEED)
    print(f"Starting battery: {reset_info['battery_level'] * 10}%")
    print(f"Departure target: {env.target_battery * 10}%")
    print(f"Charging window: {env.time_horizon} hours")
    print("\nThe agent will now make its charging decisions.")

    # Track battery and price information for the graph.
    battery_history = [int(observation[1]) * 10]
    price_history = []
    records = []

    # Track the different costs.
    total_electricity_cost = 0.0
    total_wear_penalty = 0.0
    total_shortfall_penalty = 0.0
    total_cost = 0.0

    terminated = False
    truncated = False
    final_info = None

    #Let the trained agent control the environment.
    while not (terminated or truncated):
        # Get the Q-values for the current state.
        state_index = agent.state_index(observation)
        q_values = agent.q_table[state_index].copy()

        # Choose a greedy action.
        # Epsilon = 0 means no exploration.
        action = agent.select_action(observation, epsilon=0.0)

        # Execute the chosen action.
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Collect cost information.
        total_electricity_cost += info["electricity_cost"]
        total_wear_penalty += info["wear_penalty"]
        total_shortfall_penalty += info["shortfall_penalty"]
        total_cost += info["total_cost"]

        # Store battery and electricity price for the graph.
        battery_history.append(info["battery_after"] * 10)
        price_history.append(info["price_per_kwh"])

        # Save the decision and its Q-values.
        record = {
            "hour": info["hour"] - 1,
            "price_name": info["price_name_used"],
            "price_per_kwh": info["price_per_kwh"],
            "battery_before_percent": info["battery_before"] * 10,
            "battery_after_percent": info["battery_after"] * 10,
            "action": info["action_name"],
            "q_idle": float(q_values[0]),
            "q_slow_charge": float(q_values[1]),
            "q_fast_charge": float(q_values[2]),
            "electricity_cost": info["electricity_cost"],
            "wear_penalty": info["wear_penalty"],
            "shortfall_penalty": info["shortfall_penalty"],
            "total_cost": info["total_cost"],
        }

        records.append(record)

        # Display the agent's decision.
        print("\n" + "-" * 60)
        print(f"Hour {record['hour']:02d} | Price: {info['price_name_used']} (${info['price_per_kwh']:.2f}/kWh)")
        print(f"Battery: {info['battery_before'] * 10}% -> {info['battery_after'] * 10}%")
        print(f"Agent action: {info['action_name']}")
        print(f"Q-values: Idle={q_values[0]:.3f} | Slow={q_values[1]:.3f} | Fast={q_values[2]:.3f}")
        print(f"Electricity cost: ${info['electricity_cost']:.2f}")
        print(f"Wear penalty: ${info['wear_penalty']:.2f}")

        if info["shortfall_penalty"] > 0:
            print(f"Departure penalty: ${info['shortfall_penalty']:.2f}")

        print(f"Cost this hour: ${info['total_cost']:.2f}")
        print("-" * 60)

        # Brief pause to make the episode easier to demonstrate.
        time.sleep(STEP_DELAY_SECONDS)

        observation = next_observation
        final_info = info

    #Display final episode results.

    final_battery = final_info["battery_after"] * 10
    target_battery = final_info["target_battery"] * 10
    success = final_info["battery_after"] >= final_info["target_battery"]

    print("\n" + "=" * 60)
    print("                  EPISODE RESULTS")
    print("=" * 60)
    print(f"Final battery: {final_battery}%")
    print(f"Departure target: {target_battery}%")
    print(f"Departure success: {'YES' if success else 'NO'}")
    print(f"Total electricity cost: ${total_electricity_cost:.2f}")
    print(f"Total battery wear penalty: ${total_wear_penalty:.2f}")
    print(f"Total departure penalty: ${total_shortfall_penalty:.2f}")

    print("-" * 60)
    print(f"TOTAL COST: ${total_cost:.2f}")
    print("=" * 60)

    #Save episode data and generate the graph.
    csv_path = save_demo_csv(records)
    figure = create_demo_figure(battery_history, price_history)

    print(f"\nSaved episode data: {csv_path}")
    print("\nDemonstration completed.")
    env.close()

    # Show the graph.
    plt.show()


if __name__ == "__main__":
    main()