"""Run one reproducible episode and save a visualization and text log."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ev_charging_env import EVChargingEnv


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    env = EVChargingEnv()
    observation, reset_info = env.reset(seed=7)

    lines = [
        "EV CHARGING ENVIRONMENT DEMONSTRATION",
        "======================================",
        f"Initial observation [hour, battery level, price level]: {observation.tolist()}",
        (
            f"Initial state: hour={reset_info['hour']}, "
            f"battery={reset_info['battery_level'] * 10}%, "
            f"price={reset_info['price_name']} "
            f"(${reset_info['price_per_kwh']:.2f}/kWh)"
        ),
        "",
    ]

    # A fixed sequence makes the demonstration repeatable. It is a hand-
    # selected test sequence, not a trained reinforcement-learning policy.
    actions = [1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0]

    hours = [0]
    battery_levels = [int(observation[1])]
    prices_for_hours = [reset_info["price_per_kwh"]]
    total_reward = 0.0

    for step_number, action in enumerate(actions, start=1):
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        line = (
            f"Step {step_number:02d} | {info['action_name']:<12} | "
            f"battery {info['battery_before'] * 10:>3}% -> "
            f"{info['battery_after'] * 10:>3}% | "
            f"price used ${info['price_per_kwh']:.2f}/kWh | "
            f"energy {info['energy_added_kwh']:.1f} kWh | "
            f"electricity ${info['electricity_cost']:.2f} | "
            f"wear ${info['wear_penalty']:.2f} | "
            f"departure penalty ${info['shortfall_penalty']:.2f} | "
            f"reward {reward:.2f}"
        )
        lines.append(line)

        hours.append(int(next_observation[0]))
        battery_levels.append(int(next_observation[1]))
        if not terminated:
            prices_for_hours.append(env.price_levels[int(next_observation[2])])
        else:
            prices_for_hours.append(info["price_per_kwh"])

        if terminated or truncated:
            lines.append("")
            lines.append(
                f"Episode ended: terminated={terminated}, truncated={truncated}"
            )
            lines.append(
                f"Final battery: {info['battery_after'] * 10}% "
                f"(required: {info['target_battery'] * 10}%)"
            )
            lines.append(f"Total reward: {total_reward:.2f}")
            lines.append(f"Total cost including penalties: {-total_reward:.2f}")
            break

    output_text = "\n".join(lines)
    print(output_text)
    (OUTPUTS_DIR / "demo_output.txt").write_text(output_text, encoding="utf-8")

    # Figure 1: state trajectory from the demonstration episode.
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(hours, [level * 10 for level in battery_levels], where="post", marker="o")
    axes[0].axhline(env.target_battery * 10, linestyle="--", label="Departure target (80%)")
    axes[0].set_ylabel("Battery level (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("EV Charging Environment: One 14-Hour Episode")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(hours, prices_for_hours, where="post", marker="o")
    axes[1].set_ylabel("Electricity price ($/kWh)")
    axes[1].set_xlabel("Hour of charging window")
    axes[1].set_xticks(range(0, env.time_horizon + 1))
    axes[1].set_ylim(0, 0.40)
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "figure1_environment.png", dpi=180)
    plt.close(figure)
    print(f"\nSaved visualization: {FIGURES_DIR / 'figure1_environment.png'}")
    print(f"Saved text output: {OUTPUTS_DIR / 'demo_output.txt'}")
    env.close()


if __name__ == "__main__":
    main()
