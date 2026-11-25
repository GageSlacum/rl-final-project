# Cost-Aware EV Charging Environment

This project uses reinforcement learning to study how an electric vehicle can charge before a departure deadline while electricity prices change over time. The project includes a custom Gymnasium environment, a tabular Q-learning agent, three baseline charging strategies, and an experiment that tests different battery wear penalties.

The goal is to reach the required battery level while keeping the total charging cost low.

## Requirements

- Python 3.10 or newer
- pip
- The packages listed in `requirements.txt`

## Setup on Windows

1. Open the project folder in Visual Studio Code.
2. Open **Terminal > New Terminal**.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

4. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this command and try again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

5. Install the required packages:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

Run the commands below from the project folder with the virtual environment activated.

## Project Overview

The environment simulates a 14-hour charging period. The vehicle starts with 40% battery and must reach at least 80% before departure. Electricity prices can be low, medium, or high, and the agent does not know future prices when it chooses an action.

At each hour, the agent observes:

- The current hour.
- The battery level.
- The current electricity price level.

It can choose one of three actions:

- `0`: Idle (do not charge).
- `1`: Slow charge (add 10% battery, if capacity allows).
- `2`: Fast charge (add 20% battery, if capacity allows).

The environment limits the battery to 100%. It charges for electricity use, applies a wear penalty when fast charging adds energy, and applies a larger penalty if the vehicle leaves below the 80% target.

## How to Run the Environment Demonstration

Run:

```powershell
python run_demo.py
```

This runs a fixed sequence of charging actions to show how the environment works. The actions are selected in the script ahead of time; this is **not** the trained Q-learning agent.

The demonstration prints the battery changes, electricity prices, costs, penalties, and final result. It also creates:

- `figures/figure1_environment.png`: A graph of battery level and electricity price during the example episode.
- `outputs/demo_output.txt`: A text copy of the demonstration output.

## How to Train and Evaluate the Q-Learning Agent

Run:

```powershell
python train_q_learning.py
```

This script trains a tabular Q-learning agent for 10,000 episodes. During training, the agent interacts with the environment, receives a reward based on the costs, and updates its Q-table. The agent uses an epsilon-greedy strategy: it explores different actions early in training and gradually relies more on the actions with the highest learned Q-values.

The main training settings are:

- Training episodes: 10,000
- Learning rate: 0.10
- Discount factor: 0.95
- Starting exploration rate (epsilon): 1.00
- Minimum exploration rate: 0.05
- Exploration decay: 0.9995
- Training seed: 42

After training, the script saves the learned Q-table and evaluates four strategies over 500 episodes each, using the same set of environment seeds:

1. Q-learning agent.
2. Random baseline.
3. Immediate charging, which uses slow charging until the vehicle reaches 80%.
4. Price-threshold charging, which charges when the price is low and also charges when it can no longer wait and still reach the target.

The script creates or updates these files:

- `outputs/trained_q_table.npy`: The trained Q-table used by the agent.
- `outputs/training_metrics.csv`: Training measurements.
- `outputs/evaluation_results.csv`: Episode-level results for the four strategies.
- `outputs/evaluation_summary.txt`: Average results for each strategy.
- `figures/figure3_training_curve.png`: The Q-learning training curve.
- `figures/figure4_baseline_comparison.png`: The comparison of average cost and departure success for the four strategies.

### Main Evaluation Results

The table below shows the results from 500 evaluation episodes per strategy. Average cost includes electricity cost, battery wear penalty, and any departure shortfall penalty.

| Strategy | Average total cost | Cost standard deviation | Departure success | Average final battery |
|---|---:|---:|---:|---:|
| Q-learning | **$3.42** | $1.33 | 100% | 80% |
| Random baseline | $8.12 | $2.49 | 100% | 100% |
| Immediate charging | $4.54 | $1.76 | 100% | 80% |
| Price-threshold | $3.67 | $1.73 | 100% | 80% |

Q-learning had the lowest average total cost in this evaluation. Its average cost was about 57.9% lower than the random baseline, 24.7% lower than immediate charging, and 6.8% lower than the price-threshold strategy. All four strategies reached the required 80% departure battery level in the 500 evaluation episodes.

These are results from this simulation and these evaluation settings. They are not a guarantee of performance with real vehicles or real electricity plans.

## How to Run the Battery Wear-Penalty Experiment

Run:

```powershell
python run_wear_experiment.py
```

This script trains a separate Q-learning agent for each of three fast-charging wear penalty settings:

- `$0.00`: No fast-charging wear penalty.
- `$0.50`: The default wear penalty.
- `$1.00`: A higher wear penalty.

Each agent is trained for 10,000 episodes and evaluated over 500 episodes. The other environment settings are kept the same. This experiment tests whether the cost of battery wear changes how often the agent uses fast charging.

The script creates or updates:

- `outputs/wear_penalty_experiment.csv`: Episode-level evaluation results.
- `outputs/wear_penalty_summary.csv`: Average results for each wear penalty.
- `outputs/wear_penalty_training_metrics.csv`: Training measurements for all three conditions.
- `outputs/trained_q_table_wear_0.00.npy`: Q-table trained with no wear penalty.
- `outputs/trained_q_table_wear_0.50.npy`: Q-table trained with the default wear penalty.
- `outputs/trained_q_table_wear_1.00.npy`: Q-table trained with the higher wear penalty.
- `figures/figure5_wear_penalty_experiment.png`: Average total cost and average fast-charge actions for the three conditions.

### Wear-Penalty Experiment Results

| Fast-charge wear penalty | Average total cost | Average fast-charge actions | Departure success |
|---:|---:|---:|---:|
| $0.00 | $3.05 | 1.964 | 100% |
| $0.50 | $3.42 | 0.074 | 100% |
| $1.00 | $3.40 | 0.000 | 100% |

The agent used fast charging more often when there was no wear penalty. As the penalty increased, the agent used fast charging less often. All three agents maintained 100% departure success in these evaluations.

## How to Run the Trained-Agent Demonstration

Run:

```powershell
python run_trained_agent_demo.py
```

This script loads `outputs/trained_q_table.npy`, which is created by `train_q_learning.py`. It does not train the agent again. Instead, it runs one 14-hour episode and lets the trained agent choose an action from its learned Q-table at each hour.

The demonstration uses a fixed environment seed so that the same price sequence and decisions can be reproduced. The script displays the price, battery level, selected action, Q-values, and costs for each hour. It then reports the final battery level and total episode cost.

The script creates:

- `outputs/demo_episode.csv`: The agent's hourly decisions, Q-values, battery levels, prices, and costs.
- `figures/demo_episode.png`: A graph of battery level and electricity price during the demonstration.

The single demonstration episode is separate from the 500-episode average evaluation. Its cost may therefore be higher or lower than the average Q-learning cost reported above.

## How to Run the Automated Tests

Run:

```powershell
python -m pytest -v
```

The tests check environment initialization, reproducibility when a seed is reused, slow and fast charging, battery capacity limits, departure penalties, episode termination, and price-transition probabilities.

The current test suite contains eight tests.

## How the Environment Works

### Observation

The observation contains three values:

- Hour: The current hour, from 0 through 14.
- Battery level: An integer from 0 through 10. Each level represents 10% battery.
- Price level: `0` is low, `1` is medium, and `2` is high.

There are 14 × 11 × 3 = 462 nonterminal state combinations. The terminal hour is also represented in the observation space.

### Actions

- `0`: Idle.
- `1`: Slow charge, requesting one battery level (10%).
- `2`: Fast charge, requesting two battery levels (20%).

The battery cannot exceed its maximum capacity of 100%.

### Reward and Costs

The reward is the negative of the costs for the current step:

```text
reward = -(electricity cost + wear penalty + departure penalty)
```

The battery capacity is represented as 60 kWh, so each 10% battery level represents 6 kWh. Electricity cost depends on the energy added and the current price. Fast charging applies a $0.50 wear penalty when it successfully adds energy. At departure, the environment applies a $5 penalty for each 10% battery level below the 80% target.

### Electricity Prices

The simulated electricity prices are:

- Low: $0.10 per kWh.
- Medium: $0.20 per kWh.
- High: $0.35 per kWh.

The starting price is selected randomly. After each action, the next price is selected using probability tables. A different transition table is used during the final four hours to represent a simplified morning price peak. The agent does not know future prices when it selects an action.

## Simulation Settings

- Charging period: 14 hours.
- Starting battery: 40%.
- Required battery at departure: 80%.
- Maximum battery: 100%.
- Battery capacity: 60 kWh.
- Low electricity price: $0.10 per kWh.
- Medium electricity price: $0.20 per kWh.
- High electricity price: $0.35 per kWh.
- Fast-charge wear penalty: $0.50 for each fast-charge action that adds energy.
- Departure shortfall penalty: $5 for each missing 10% battery level.

These values are simulation assumptions. They do not represent prices or measurements from a specific utility or electric vehicle.

## Main Project Files

- `ev_charging_env.py`: Defines the custom Gymnasium charging environment.
- `run_demo.py`: Runs a fixed sequence of actions to demonstrate the environment.
- `train_q_learning.py`: Trains the Q-learning agent, evaluates the four charging strategies, and creates the training and baseline comparison results.
- `run_wear_experiment.py`: Trains and evaluates Q-learning agents under three fast-charging wear penalty settings.
- `run_trained_agent_demo.py`: Loads the saved Q-table and demonstrates the trained agent in one episode.
- `tests/test_ev_charging_env.py`: Contains the automated environment tests.
- `requirements.txt`: Lists the required Python packages.
- `figures/`: Contains the environment, training, comparison, wear-penalty, code, and demonstration figures.
- `outputs/`: Contains the saved Q-table, training data, evaluation results, experiment summaries, and demonstration data.

## Limitations

This project is a simplified simulation. It uses three electricity price levels instead of real utility pricing data, a fixed 14-hour charging period, a fixed starting battery level, and a fixed 80% departure target. It does not model a real vehicle, charger, or electricity provider.

The main project uses tabular Q-learning and one training seed for the reported results. The evaluation uses 500 environment seeds per strategy, but the agent was not retrained across multiple independent training seeds. SARSA, value iteration, and more advanced reinforcement-learning algorithms were not implemented. Unexpected departure times and changing battery requirements were also left as possible future improvements.
