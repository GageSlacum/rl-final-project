# COST-AWARE EV CHARGING ENVIRONMENT

This project contains a custom Gymnasium environment that simulates charging an electric vehicle while electricity prices change over a 14-hour period. It also includes a tabular Q-learning agent that learns charging decisions to reduce costs while meeting the required battery level.

## WHAT YOU NEED

- Python 3.10 or newer
- pip
- An internet connection for installing the required packages

## HOW TO SET IT UP ON WINDOWS

1. Open the project folder in Visual Studio Code.
2. Open Terminal, then select New Terminal.
3. Create a virtual environment by entering:

   ```powershell
   py -m venv .venv
   ```

4. Turn on the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks this command, enter this command first, then try activating the environment again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

5. Install the project packages:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## HOW TO RUN THE DEMONSTRATION

Enter this command in the terminal:

```powershell
python run_demo.py
```

The demonstration runs a fixed 14-hour sequence of charging actions. It prints the actions, battery changes, electricity costs, penalties, and final result.

It also creates these files:

- `figures/figure1_environment.png`: A chart of battery level and electricity price during the episode.
- `outputs/demo_output.txt`: A text copy of the demonstration output.

The demonstration uses actions chosen ahead of time. It is separate from the trained reinforcement-learning agent.

## HOW TO TRAIN THE Q-LEARNING AGENT

Enter this command in the terminal:

```powershell
python train_q_learning.py
```

This runs the Q-learning training process. The agent learns by interacting with the charging environment over 10,000 episodes.

During training, the agent observes the current hour, battery level, and electricity price. It selects a charging action, receives a reward, and updates its Q-table based on the result.

The training process uses the following parameters:

- Training episodes: 10,000
- Learning rate: 0.10
- Discount factor: 0.95
- Starting exploration rate: 1.00
- Minimum exploration rate: 0.05
- Exploration gradually decreases during training.

The agent uses an epsilon-greedy strategy. This means it sometimes chooses random actions to explore different charging decisions and otherwise chooses the action with the highest estimated Q-value.

After training, the program evaluates the learned policy against a random baseline over 500 episodes.

The program also generates training graphs and saves the training and evaluation results.

## INITIAL TRAINING RESULTS

The Q-learning agent was evaluated over 500 episodes and compared against a random policy using the same set of environment seeds.

The initial results were:

| Metric | Q-learning | Random baseline |
|---|---:|---:|
| Average total cost | $3.42 | $8.12 |
| Cost standard deviation | $1.33 | $2.49 |
| Departure success rate | 100% | 100% |
| Average final battery | 80% | 100% |

The Q-learning agent achieved approximately 57.9% lower average total cost than the random baseline while maintaining a 100% departure success rate.

The results suggest that the agent learned to avoid unnecessary charging while still reaching the required battery level.

## HOW TO RUN THE TESTS

Enter this command:

```powershell
python -m pytest -q
```

The tests check that the environment resets correctly, produces the same starting state when a seed is reused, handles slow and fast charging, keeps the battery within its limit, ends at the correct time, applies the departure penalty, and uses valid price-transition probabilities.

## HOW THE ENVIRONMENT WORKS

### OBSERVATION

The environment provides three numbers:

- Hour: The current hour, from 0 to 14.
- Battery level: A number from 0 to 10. Each level represents 10% battery.
- Price level: 0 means low, 1 means medium, and 2 means high electricity price.

There are 14 x 11 x 3 = 462 possible nonterminal combinations of these values. The final observation is at hour 14, after the last charging action.

### ACTIONS

- 0: Do not charge.
- 1: Slow charge. This requests one battery level, or 10%.
- 2: Fast charge. This requests two battery levels, or 20%.

The battery cannot go above 100%.

### REWARD

The reward is the negative of the costs for that hour:

```text
reward = -(electricity cost + wear penalty + departure penalty)
```

Each battery level represents 6 kWh. Electricity cost depends on how much energy is added and the current price. Fast charging adds a $0.50 wear penalty when it adds energy.

At the end of the 14-hour period, the environment applies a $5 penalty for every 10% battery level below the 80% departure target.

### ELECTRICITY PRICES

The starting price is randomly selected from low, medium, and high prices.

After each action, the next price is selected using probability tables. A different table is used during the final four hours to make medium and high prices more likely, representing a simplified morning price peak.

## SIMULATION SETTINGS

- Charging period: 14 hours
- Starting battery: 40%
- Battery target at departure: 80%
- Maximum battery: 100%
- Battery capacity represented: 60 kWh
- Low electricity price: $0.10 per kWh
- Medium electricity price: $0.20 per kWh
- High electricity price: $0.35 per kWh
- Fast-charge wear penalty: $0.50 per fast-charge action that adds energy
- Departure shortfall penalty: $5 for each missing 10% battery level

These values are simulation assumptions for the project. They are not actual prices or measurements from a specific utility or vehicle.

## MAIN PROJECT FILES

- `ev_charging_env.py`: The custom charging environment.
- `run_demo.py`: Runs the demonstration and saves the chart and output log.
- `train_q_learning.py`: Implements and trains the tabular Q-learning agent, evaluates it against a random baseline, and generates results.
- `tests/test_ev_charging_env.py`: The automated environment tests.
- `requirements.txt`: The Python packages needed by the project.
- `figures/`: Contains generated environment, training, and comparison graphs.
- `outputs/`: Contains saved training and evaluation results.

## LIMITATIONS AND NEXT STEPS

The current environment uses three electricity price levels, fixed starting and target battery levels, and a fixed 14-hour charging period.

The current implementation uses tabular Q-learning and a random policy for its initial baseline comparison. SARSA and more advanced reinforcement-learning algorithms have not yet been implemented.

Future improvements include testing different learning rates and exploration settings, introducing additional electricity price patterns, and investigating unexpected departure times and varying battery requirements.
