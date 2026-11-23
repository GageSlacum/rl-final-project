"""Train and evaluate a tabular Q-learning agent for EV charging."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ev_charging_env import EVChargingEnv

# Experiment settings
ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAINING_EPISODES = 10000
EVALUATION_EPISODES = 500
TRAINING_SEED = 42
EVALUATION_SEED_START = 10000

LEARNING_RATE = 0.10
DISCOUNT_FACTOR = 0.95
STARTING_EPSILON = 1.00
MINIMUM_EPSILON = 0.05
EPSILON_DECAY = 0.9995
MOVING_AVERAGE_WINDOW = 200


# Q-learning agent
class QLearningAgent:
    """A simple Q-learning agent with a table of state-action values."""

    def __init__(self, env, seed=TRAINING_SEED):
        state_dimensions = tuple(int(value) for value in env.observation_space.nvec)
        number_of_actions = int(env.action_space.n)

        # The table stores one Q-value for every state and possible action.
        self.q_table = np.zeros(
            state_dimensions + (number_of_actions,), dtype=np.float64
        )
        self.rng = np.random.default_rng(seed)

    def state_index(self, observation):
        """Convert [hour, battery, price] into a tuple for indexing the table."""
        return tuple(int(value) for value in observation)

    def select_action(self, observation, epsilon):
        """Choose a random action sometimes; otherwise choose the best known action."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q_table.shape[-1]))

        state = self.state_index(observation)
        action_values = self.q_table[state]
        best_value = np.max(action_values)
        best_actions = np.flatnonzero(np.isclose(action_values, best_value))

        # Randomly break ties so the agent does not always prefer action 0.
        return int(self.rng.choice(best_actions))

    def update(self, observation, action, reward, next_observation, episode_done):
        """Apply the Q-learning update after one environment step."""
        state = self.state_index(observation)
        next_state = self.state_index(next_observation)
        old_value = self.q_table[state + (action,)]

        # If the episode ended, there is no future reward to add.
        if episode_done:
            target = reward
        else:
            best_future_value = np.max(self.q_table[next_state])
            target = reward + DISCOUNT_FACTOR * best_future_value

        self.q_table[state + (action,)] = old_value + LEARNING_RATE * (
            target - old_value
        )


# Training
def train_agent(env, agent):
    """Let the agent practice for many complete charging episodes."""
    training_records = []
    epsilon = STARTING_EPSILON

    for episode in range(1, TRAINING_EPISODES + 1):
        if episode == 1:
            observation, _ = env.reset(seed=TRAINING_SEED)
        else:
            observation, _ = env.reset()

        total_reward = 0.0
        terminated = False
        truncated = False
        final_info = None

        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated

            agent.update(
                observation,
                action,
                reward,
                next_observation,
                episode_done,
            )

            total_reward += reward
            observation = next_observation
            final_info = info

        success = final_info["battery_after"] >= final_info["target_battery"]
        episode_cost = -total_reward

        training_records.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "total_cost": episode_cost,
                "success": int(success),
                # Each battery level represents 10 percent.
                "final_battery_percent": final_info["battery_after"] * 10,
                "epsilon": epsilon,
            }
        )

        # Slowly reduce exploration, but never let it fall below the minimum.
        epsilon = max(MINIMUM_EPSILON, epsilon * EPSILON_DECAY)

        if episode % 1000 == 0:
            recent = training_records[-100:]
            average_cost = np.mean([row["total_cost"] for row in recent])
            average_success = np.mean([row["success"] for row in recent]) * 100
            print(
                f"Episode {episode:5d}/{TRAINING_EPISODES} | "
                f"last-100 average cost: ${average_cost:.2f} | "
                f"success rate: {average_success:.1f}% | "
                f"epsilon: {epsilon:.3f}"
            )

    return training_records

# Evaluation: Q-learning and three baseline policies

def select_q_learning_action(observation, env, agent, rng):
    """Select the best action from the trained Q-learning agent."""
    return agent.select_action(observation, epsilon=0.0)


def select_random_action(observation, env, agent, rng):
    """Select a random charging action."""
    return int(rng.integers(env.action_space.n))


def select_immediate_charging_action(observation, env, agent, rng):
    """
    Charge immediately using slow charging until the target is reached.
    Then remain idle.
    """
    battery_level = int(observation[1])

    if battery_level < env.target_battery:
        return env.SLOW_CHARGE

    return env.IDLE


def select_price_threshold_action(observation, env, agent, rng):
    """
    Charge when electricity is cheap.
    If waiting would prevent reaching the target, charge regardless of price.
    """
    hour = int(observation[0])
    battery_level = int(observation[1])
    price_level = int(observation[2])

    # Calculate how much charging is still required.
    required_levels = max(0, env.target_battery - battery_level)

    # Calculate how many hours remain before departure.
    remaining_hours = env.time_horizon - hour

    # Stop charging once the target has been reached.
    if required_levels == 0:
        return env.IDLE

    # Do not wait if every remaining hour is needed to reach the target.
    if remaining_hours <= required_levels:
        return env.SLOW_CHARGE

    # Price level 0 represents low electricity prices.
    if price_level == 0:
        return env.SLOW_CHARGE

    # Otherwise, wait for a cheaper price.
    return env.IDLE


def evaluate_policy(env, policy_name, action_selector, agent=None):
    """Evaluate one charging policy over the same set of environment seeds."""

    records = []

    # This random generator is used by the random baseline.
    random_generator = np.random.default_rng(2026)

    for episode in range(EVALUATION_EPISODES):

        episode_seed = EVALUATION_SEED_START + episode

        observation, _ = env.reset(seed=episode_seed)

        total_cost = 0.0
        electricity_cost = 0.0
        wear_penalty = 0.0
        shortfall_penalty = 0.0
        fast_charge_actions = 0

        terminated = False
        truncated = False
        final_info = None

        while not (terminated or truncated):

            # Ask the selected policy which action it wants to take.
            action = action_selector(
                observation, env, agent, random_generator
            )

            observation, _, terminated, truncated, info = env.step(action)

            # Record the different parts of the charging cost.
            total_cost += info["total_cost"]
            electricity_cost += info["electricity_cost"]
            wear_penalty += info["wear_penalty"]
            shortfall_penalty += info["shortfall_penalty"]

            # Count fast charges that actually add battery energy.
            if action == env.FAST_CHARGE and info["levels_added"] > 0:
                fast_charge_actions += 1

            final_info = info

        records.append(
            {
                "policy": policy_name,
                "episode": episode + 1,
                "total_cost": total_cost,
                "electricity_cost": electricity_cost,
                "wear_penalty": wear_penalty,
                "shortfall_penalty": shortfall_penalty,
                "fast_charge_actions": fast_charge_actions,
                "success": int(
                    final_info["battery_after"] >= final_info["target_battery"]
                ),
                "final_battery_percent": final_info["battery_after"] * 10,
            }
        )

    return records

# Save results and figures
def save_training_csv(records):
    path = OUTPUTS_DIR / "training_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def save_evaluation_csv(records):
    path = OUTPUTS_DIR / "evaluation_results.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def summarize_policy(records):

    costs = np.array(
        [row["total_cost"] for row in records], dtype=float
    )

    success_rate = np.mean(
        [row["success"] for row in records]
    ) * 100

    final_battery = np.mean(
        [row["final_battery_percent"] for row in records]
    )

    return {
        "average_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "success_rate": float(success_rate),
        "average_final_battery_percent": float(final_battery),

        "average_electricity_cost": float(
            np.mean([row["electricity_cost"] for row in records])
        ),

        "average_wear_penalty": float(
            np.mean([row["wear_penalty"] for row in records])
        ),

        "average_shortfall_penalty": float(
            np.mean([row["shortfall_penalty"] for row in records])
        ),

        "average_fast_charge_actions": float(
            np.mean([row["fast_charge_actions"] for row in records])
        ),
    }


def save_training_figure(records):
    costs = np.array([row["total_cost"] for row in records], dtype=float)
    successes = np.array([row["success"] for row in records], dtype=float)
    window = MOVING_AVERAGE_WINDOW
    averages = np.ones(window) / window

    smoothed_costs = np.convolve(costs, averages, mode="valid")
    smoothed_success = np.convolve(successes, averages, mode="valid") * 100
    episode_numbers = np.arange(window, len(records) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(episode_numbers, smoothed_costs)
    axes[0].set_ylabel("Average episode cost ($)")
    axes[0].set_title("Q-Learning Training Progress (200-Episode Moving Average)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episode_numbers, smoothed_success)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "figure3_training_curve.png", dpi=180)
    plt.close(figure)

def save_baseline_figure(policy_records):

    summaries = {
        policy_name: summarize_policy(records)
        for policy_name, records in policy_records.items()
    }

    labels = list(summaries.keys())

    average_costs = [
        summaries[label]["average_cost"] for label in labels
    ]

    success_rates = [
        summaries[label]["success_rate"] for label in labels
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compare average total charging costs.
    axes[0].bar(labels, average_costs)
    axes[0].set_title("Average Total Cost")
    axes[0].set_ylabel("Cost per episode ($)")
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    # Compare the percentage of successful departures.
    axes[1].bar(labels, success_rates)
    axes[1].set_title("Departure Success Rate")
    axes[1].set_ylabel("Episodes reaching 80% battery (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("Q-Learning Compared with Charging Baselines")

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "figure4_baseline_comparison.png",
        dpi=180
    )

    plt.close(figure)

    return summaries

def save_summary_file(summaries):

    path = OUTPUTS_DIR / "evaluation_summary.txt"

    with path.open("w", encoding="utf-8") as output_file:

        output_file.write(
            f"Evaluation episodes per policy: {EVALUATION_EPISODES}\n"
        )

        output_file.write(
            "All policies use the same set of environment seeds.\n\n"
        )

        for policy_name, values in summaries.items():

            output_file.write(f"{policy_name}\n")

            output_file.write(f"Average total cost: ${values['average_cost']:.2f}\n")
            output_file.write(f"Cost standard deviation: ${values['cost_std']:.2f}\n")
            output_file.write(f"Electricity cost: ${values['average_electricity_cost']:.2f}\n")
            output_file.write(f"Wear penalty: ${values['average_wear_penalty']:.2f}\n")
            output_file.write(f"Departure penalty: ${values['average_shortfall_penalty']:.2f}\n")
            output_file.write(f"Average fast-charge actions: {values['average_fast_charge_actions']:.2f}\n")
            output_file.write(f"Departure success rate: {values['success_rate']:.1f}%\n")
            output_file.write(f"Average final battery: {values['average_final_battery_percent']:.1f}%\n\n")


def main():
    env = EVChargingEnv()
    agent = QLearningAgent(env)
    print("Training the Q-learning agent...")
    training_records = train_agent(env, agent)
    save_training_csv(training_records)
    np.save(OUTPUTS_DIR / "trained_q_table.npy", agent.q_table)
    save_training_figure(training_records)

    # Define the four policies that will be evaluated.
    policy_definitions = [
        ("Q-learning", select_q_learning_action, agent),
        ("Random baseline", select_random_action, None),
        ("Immediate charging", select_immediate_charging_action, None),
        ("Price-threshold", select_price_threshold_action, None),
    ]
    policy_records = {}
    all_evaluation_records = []

    # Evaluate each policy using the same environment seeds.
    for policy_name, action_selector, policy_agent in policy_definitions:
        print(f"\nEvaluating {policy_name}...")
        records = evaluate_policy(
            env,
            policy_name,
            action_selector,
            agent=policy_agent
        )
        policy_records[policy_name] = records
        all_evaluation_records.extend(records)

    # Save all evaluation results.
    save_evaluation_csv(all_evaluation_records)

    # Generate the comparison graph and calculate summaries.
    summaries = save_baseline_figure(policy_records)
    save_summary_file(summaries)
    print("\nFINAL EVALUATION RESULTS")

    for policy_name, values in summaries.items():
        print(f"\n{policy_name}")
        print(f"  Average total cost: ${values['average_cost']:.2f}")
        print(f"  Electricity cost: ${values['average_electricity_cost']:.2f}")
        print(f"  Wear penalty: ${values['average_wear_penalty']:.2f}")
        print(f"  Departure penalty: ${values['average_shortfall_penalty']:.2f}")
        print(f"  Average fast-charge actions: {values['average_fast_charge_actions']:.2f}")
        print(f"  Departure success rate: {values['success_rate']:.1f}%")
        print(f"  Average final battery: {values['average_final_battery_percent']:.1f}%")

    print("\nCreated files:")
    print("  figures/figure3_training_curve.png")
    print("  figures/figure4_baseline_comparison.png")
    print("  outputs/training_metrics.csv")
    print("  outputs/evaluation_results.csv")
    print("  outputs/evaluation_summary.txt")
    print("  outputs/trained_q_table.npy")
    env.close()


if __name__ == "__main__":
    main()
