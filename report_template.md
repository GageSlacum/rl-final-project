# Milestone 2: Environment Setup

**Project:** Cost-Aware Electric Vehicle Charging Under Uncertain Electricity Prices  
**Student:** Gage Slacum  
**Date:** November 3, 2026

## 1. Project Overview

- This project investigates how reinforcement learning can schedule electric vehicle charging over a 14-hour overnight window when electricity prices change over time.
- I implemented a custom Gymnasium environment with three charging actions, a discrete battery state, stochastic electricity prices, charging costs, a fast-charging wear penalty, and a departure deadline penalty.
- The next stage is to implement Q-learning and SARSA, then compare their results with simple charging baselines.

## 2. Environment Design

**Environment:** The environment simulates a 60 kWh electric vehicle battery over a 14-hour charging window. The initial battery is 40%, the required departure battery is 80%, and the battery cannot exceed 100%.

**State/Observation:** The agent receives `[hour, battery_level, price_level]`. Hour is 0–13 during decision-making (14 is the terminal observation), battery level is 0–10 in 10% increments, and price level is 0 (low), 1 (medium), or 2 (high). There are 462 nonterminal decision-state combinations.

**Action:** The agent selects idle (0), slow charge (1 battery level / 10%), or fast charge (2 battery levels / 20%). Charging is capped at 100% battery.

**Reward:** The reward is the negative sum of electricity cost, fast-charging wear penalty, and departure shortfall penalty. Electricity cost equals the energy added in kWh multiplied by the current electricity price; fast charging adds a $0.50 wear penalty when it adds energy; at departure, the environment applies $5 for each 10% battery level below the 80% target.

**Transition:** After the action, the environment updates the battery, advances one hour, and samples the next price using a three-level transition model. A separate transition matrix is used during the final four hours to represent a simplified morning peak. The episode terminates after 14 actions.

**Simulation assumptions:** Low, medium, and high prices are $0.10, $0.20, and $0.35 per kWh. Each 10% battery level represents 6 kWh of a 60 kWh battery. These values are simulation assumptions for this milestone, not real utility-rate data.

**Figure 1. Environment visualization.** Insert `figures/figure1_environment.png` here. The figure shows the battery trajectory and electricity price over one 14-hour demonstration episode. The dashed horizontal line represents the 80% departure target.

**Figure 2. Environment implementation code.** Insert a screenshot from `ev_charging_env.py` showing the `step()` method and its action, battery-capacity, reward, price-transition, and termination logic. The `step()` method applies the selected action, calculates electricity and wear costs, advances the hour, samples the next price when the episode continues, and applies the departure penalty on the final hour.

## 3. Environment Testing

- I ran the environment using a fixed random seed and a demonstration sequence of idle, slow-charge, and fast-charge actions. [Confirm after running `python run_demo.py`.]
- Each action updates the battery and advances the hour; the reward is the negative cost of electricity, battery wear, and any departure shortfall.
- I verified that `reset()` restores the initial hour and battery state, returns a valid observation, and reproduces the initial observation when the same seed is reused. [Confirm after running the tests.]
- Automated tests check slow charging, fast charging, the 100% battery limit, termination at hour 14, the departure shortfall penalty, and valid price-transition probabilities. [Replace with the actual test result.]

**Figure 3. Execution and testing evidence.** Insert a screenshot showing the terminal output from `python run_demo.py` and a separate screenshot showing the actual result from `python -m pytest -q`. The demo output shows the action-by-action battery and cost changes; the test output shows how many checks passed. Do not claim that the tests passed unless the command completes successfully on your computer.

## 4. Next Steps

- Implement tabular Q-learning and SARSA using the discrete state and action spaces, then compare their performance with random, immediate-charging, and fixed-schedule baselines.
- Experiment with alternative price patterns and wear penalties, and evaluate electricity cost, target-charge success rate, wear penalty, and expected return.
- A current limitation is that the initial implementation uses simplified price levels and fixed battery/departure parameters rather than real utility data or a learned charging policy.
