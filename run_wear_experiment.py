"""Compare Q-learning under different fast-charging wear penalties."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from ev_charging_env import EVChargingEnv

from train_q_learning import (
    TRAINING_SEED,
    QLearningAgent,
    train_agent,
    evaluate_policy,
    select_q_learning_action,
    FIGURES_DIR,
    OUTPUTS_DIR,
)


# Test three different fast-charging wear penalties.
WEAR_PENALTIES = [0.00, 0.50, 1.00]


def save_csv(path, records):
    """Save a list of dictionaries as a CSV file."""

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))

        writer.writeheader()
        writer.writerows(records)


def summarize_results(records, wear_penalty_setting):
    """Calculate the average performance for one wear condition."""

    return {
        "wear_penalty_setting": wear_penalty_setting,
        "average_total_cost": float(np.mean([r["total_cost"] for r in records])),
        "cost_std": float(np.std([r["total_cost"] for r in records])),
        "average_electricity_cost": float(np.mean([r["electricity_cost"] for r in records])),
        "average_actual_wear_penalty": float(np.mean([r["wear_penalty"] for r in records])),
        "average_shortfall_penalty": float(np.mean([r["shortfall_penalty"] for r in records])),
        "average_fast_charge_actions": float(np.mean([r["fast_charge_actions"] for r in records])),
        "success_rate": float(np.mean([r["success"] for r in records]) * 100),
        "average_final_battery_percent": float(np.mean([r["final_battery_percent"] for r in records])),
    }


def create_experiment_figure(summaries):
    """Plot the effect of wear penalties on learned charging behavior."""

    penalties = [
        row["wear_penalty_setting"] for row in summaries
    ]

    average_costs = [
        row["average_total_cost"] for row in summaries
    ]

    fast_charge_actions = [
        row["average_fast_charge_actions"] for row in summaries
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    # First graph: average total cost.
    axes[0].plot(penalties, average_costs, marker="o")
    axes[0].set_title("Average Total Cost")
    axes[0].set_xlabel("Fast-Charging Wear Penalty ($)")
    axes[0].set_ylabel("Average Cost per Episode ($)")
    axes[0].set_xticks(penalties)
    axes[0].grid(True, alpha=0.3)

    # Second graph: fast-charging behavior.
    axes[1].plot(penalties, fast_charge_actions, marker="o")
    axes[1].set_title("Fast-Charging Behavior")
    axes[1].set_xlabel("Fast-Charging Wear Penalty ($)")
    axes[1].set_ylabel("Average Fast-Charge Actions")
    axes[1].set_xticks(penalties)
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Effect of Battery Wear Penalties on Q-Learning")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "figure5_wear_penalty_experiment.png",
        dpi=180
    )
    plt.close(figure)


def main():

    all_episode_results = []
    all_training_results = []
    all_summaries = []

    for wear_penalty in WEAR_PENALTIES:

        print("\n" + "=" * 55)
        print(f"EXPERIMENT: Fast-Charging Wear Penalty = ${wear_penalty:.2f}")
        print("=" * 55)

        # Create a new environment using this wear penalty.
        env = EVChargingEnv(fast_charge_wear_penalty=wear_penalty)

        # Create a fresh Q-learning agent.
        agent = QLearningAgent(env, seed=TRAINING_SEED)

        # Train the agent from scratch.
        print("\nTraining Q-learning agent...")
        training_records = train_agent(env, agent)

        # Save this condition's training results.
        for record in training_records:
            all_training_results.append({"wear_penalty_setting": wear_penalty, **record})

        # Save this condition's trained Q-table.
        table_name = (f"trained_q_table_wear_{wear_penalty:.2f}.npy")
        np.save(OUTPUTS_DIR / table_name, agent.q_table)

        # Evaluate the trained agent on 500 episodes.
        print("\nEvaluating trained agent...")
        evaluation_records = evaluate_policy(
            env,
            "Q-learning",
            select_q_learning_action,
            agent=agent
        )

        # Identify which wear condition produced each result.
        for record in evaluation_records:
            all_episode_results.append({"wear_penalty_setting": wear_penalty, **record})

        # Calculate the average performance.
        summary = summarize_results(evaluation_records, wear_penalty)
        all_summaries.append(summary)
        env.close()
        print(f"\nResults for wear penalty ${wear_penalty:.2f}:")
        print(f"  Average total cost: ${summary['average_total_cost']:.2f}")
        print(f"  Average electricity cost: ${summary['average_electricity_cost']:.2f}")
        print(f"  Average actual wear penalty: ${summary['average_actual_wear_penalty']:.2f}")
        print(f"  Average fast-charge actions: {summary['average_fast_charge_actions']:.2f}")
        print(f"  Departure success rate: {summary['success_rate']:.1f}%")

    # Save all experiment data.
    save_csv(OUTPUTS_DIR / "wear_penalty_experiment.csv", all_episode_results)
    save_csv(OUTPUTS_DIR / "wear_penalty_summary.csv", all_summaries)
    save_csv(OUTPUTS_DIR / "wear_penalty_training_metrics.csv", all_training_results)

    # Generate Figure 5.
    create_experiment_figure(all_summaries)

    print("\n" + "=" * 55)
    print("WEAR PENALTY EXPERIMENT COMPLETED")
    print("=" * 55)

    for summary in all_summaries:
        print(f"\nWear penalty: ${summary['wear_penalty_setting']:.2f}")
        print(f"  Average cost: ${summary['average_total_cost']:.2f}")
        print(f"  Fast-charge actions: {summary['average_fast_charge_actions']:.2f}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")

    print("\nCreated files:")
    print("  figures/figure5_wear_penalty_experiment.png")
    print("  outputs/wear_penalty_experiment.csv")
    print("  outputs/wear_penalty_summary.csv")
    print("  outputs/wear_penalty_training_metrics.csv")
    print("  outputs/trained_q_table_wear_0.00.npy")
    print("  outputs/trained_q_table_wear_0.50.npy")
    print("  outputs/trained_q_table_wear_1.00.npy")


if __name__ == "__main__":
    main()