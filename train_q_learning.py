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

# Evaluation: trained agent and random baseline
def evaluate_q_learning(env, agent):
    """Test the trained policy without exploration."""
    records = []

    for episode in range(EVALUATION_EPISODES):
        episode_seed = EVALUATION_SEED_START + episode
        observation, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        final_info = None

        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_info = info

        records.append(
            {
                "policy": "Q-learning",
                "episode": episode + 1,
                "total_cost": -total_reward,
                "success": int(
                    final_info["battery_after"] >= final_info["target_battery"]
                ),
                # Each battery level represents 10 percent.
                "final_battery_percent": final_info["battery_after"] * 10,
            }
        )

    return records


def evaluate_random_baseline(env):
    """Test a policy that chooses idle, slow, or fast charging at random."""
    records = []
    random_generator = np.random.default_rng(2026)

    for episode in range(EVALUATION_EPISODES):
        episode_seed = EVALUATION_SEED_START + episode
        observation, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        final_info = None

        while not (terminated or truncated):
            action = int(random_generator.integers(env.action_space.n))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_info = info

        records.append(
            {
                "policy": "Random baseline",
                "episode": episode + 1,
                "total_cost": -total_reward,
                "success": int(
                    final_info["battery_after"] >= final_info["target_battery"]
                ),
                # Each battery level represents 10 percent.
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
    costs = np.array([row["total_cost"] for row in records], dtype=float)
    success_rate = np.mean([row["success"] for row in records]) * 100
    final_battery = np.mean(
        [row["final_battery_percent"] for row in records]
    )

    return {
        "average_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "success_rate": float(success_rate),
        "average_final_battery_percent": float(final_battery),
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


def save_baseline_figure(q_records, random_records):
    summaries = {
        "Q-learning": summarize_policy(q_records),
        "Random baseline": summarize_policy(random_records),
    }

    labels = list(summaries.keys())
    average_costs = [summaries[label]["average_cost"] for label in labels]
    success_rates = [summaries[label]["success_rate"] for label in labels]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(labels, average_costs)
    axes[0].set_title("Average Total Cost")
    axes[0].set_ylabel("Cost per episode ($)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, success_rates)
    axes[1].set_title("Departure Success Rate")
    axes[1].set_ylabel("Episodes reaching 80% battery (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("Q-Learning Compared with a Random Policy")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "figure4_baseline_comparison.png", dpi=180)
    plt.close(figure)

    return summaries


def save_summary_file(summaries):
    path = OUTPUTS_DIR / "evaluation_summary.txt"
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"Evaluation episodes per policy: {EVALUATION_EPISODES}\n")
        output_file.write("Both policies use the same set of environment seeds.\n\n")

        for policy_name, values in summaries.items():
            output_file.write(f"{policy_name}\n")
            output_file.write(f"Average total cost: ${values['average_cost']:.2f}\n")
            output_file.write(f"Cost standard deviation: ${values['cost_std']:.2f}\n")
            output_file.write(f"Departure success rate: {values['success_rate']:.1f}%\n")
            output_file.write(
                "Average final battery: "
                f"{values['average_final_battery_percent']:.1f}%\n\n"
            )


def main():
    env = EVChargingEnv()
    agent = QLearningAgent(env)

    print("Training the Q-learning agent...")
    training_records = train_agent(env, agent)
    save_training_csv(training_records)
    np.save(OUTPUTS_DIR / "trained_q_table.npy", agent.q_table)
    save_training_figure(training_records)

    print("\nEvaluating the trained agent...")
    q_records = evaluate_q_learning(env, agent)

    print("Evaluating the random baseline...")
    random_records = evaluate_random_baseline(env)

    all_evaluation_records = q_records + random_records
    save_evaluation_csv(all_evaluation_records)
    summaries = save_baseline_figure(q_records, random_records)
    save_summary_file(summaries)

    print("\nFINAL EVALUATION RESULTS")
    for policy_name, values in summaries.items():
        print(f"\n{policy_name}")
        print(f"  Average total cost: ${values['average_cost']:.2f}")
        print(f"  Departure success rate: {values['success_rate']:.1f}%")
        print(
            "  Average final battery: "
            f"{values['average_final_battery_percent']:.1f}%"
        )

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
