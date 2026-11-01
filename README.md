# Cost-Aware EV Charging Environment

A custom Gymnasium environment for the Milestone 2 implementation of the project **Cost-Aware Electric Vehicle Charging Under Uncertain Electricity Prices**.

## Requirements

- Python 3.10 or newer
- pip
- Internet access for the first installation of the Python packages

## Setup on Windows (PowerShell)

1. Open this folder in Visual Studio Code.
2. Open **Terminal > New Terminal**.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

4. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use the Command Prompt instructions below or run the one-session command:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run the demonstration

```powershell
python run_demo.py
```

The script runs one fixed, reproducible 14-hour action sequence. It prints each action, battery change, electricity cost, wear penalty, reward, and final result. It also creates:

- `figures/figure1_environment.png` — a plot of battery level and electricity price during the episode.
- `outputs/demo_output.txt` — a text copy of the demonstration output.

The demonstration sequence is a manually selected test sequence, **not a trained reinforcement-learning agent**.

## Run the automated tests

```powershell
python -m pytest -q
```

The tests cover reset behavior, seed reproducibility, slow and fast charging, battery capacity, departure termination, the departure shortfall penalty, and price-transition probabilities.

## Environment design

### Observation

A three-integer NumPy array: `[hour, battery_level, price_level]`.

- `hour`: 0 through 13 for decision steps; 14 is the terminal observation.
- `battery_level`: 0 through 10, where each level is 10% of the battery.
- `price_level`: 0 = low, 1 = medium, 2 = high.

There are 14 × 11 × 3 = 462 nonterminal decision-state combinations. The observation space also includes the terminal hour so the final observation is valid.

### Actions

- `0`: idle, add zero battery levels.
- `1`: slow charge, request one battery level (+10%).
- `2`: fast charge, request two battery levels (+20%).

The battery level is capped at 10.

### Reward

The reward is the negative of all costs incurred on that step:

`reward = -(electricity_cost + wear_penalty + departure_shortfall_penalty)`

`electricity_cost = energy_added_kwh × current_price_per_kwh`

Each battery level represents 6 kWh. Fast charging incurs a $0.50 wear penalty when it adds energy. At the end of the 14-hour window, the environment subtracts $5 for every battery level below the 80% target.

### Transitions

After the agent acts, the environment applies the charge, advances the hour, and samples the next electricity price from a three-state transition matrix. During the final four hours, a second transition matrix gives medium/high prices a greater chance to occur, representing a simplified morning peak. At hour 14, the episode terminates and the departure shortfall penalty is applied.

## Parameters used for this milestone

These are explicit simulation assumptions selected to make the proposal executable; they are not measurements from a particular utility or EV:

| Parameter | Value |
|---|---:|
| Charging window | 14 hours |
| Starting battery | 40% |
| Departure target | 80% |
| Battery capacity | 100% |
| Battery capacity represented | 60 kWh |
| Low / medium / high price | $0.10 / $0.20 / $0.35 per kWh |
| Fast-charge wear penalty | $0.50 per fast-charge action that adds energy |
| Departure shortfall penalty | $5 per missing 10% battery level |

## Project files

- `ev_charging_env.py`: custom Gymnasium environment.
- `run_demo.py`: runs a demonstration and saves the plot and output log.
- `tests/test_ev_charging_env.py`: automated tests.
- `Report_Draft.docx`: editable two-page report draft with figure placeholders. Replace the placeholders and update the testing statements before exporting your final PDF.
- `report_template.md`: plain-text/Markdown version of the report draft.
- `requirements.txt`: Python dependencies.

## Limitations and next steps

This milestone uses a simplified three-level price model and fixed initial/target battery levels. No Q-learning, SARSA, policy training, or baseline comparison is included yet. Those are planned for the next milestone, along with experiments using different price patterns and possibly unexpected departure times.
