# Fleet Battery Optimization 

This repository contains a Python-based optimization model that demonstrates how to schedule a fleet of batteries across multiple markets and time steps to maximize total revenue. The problem formulation and code originate from a trading engineer home assignment, focusing on the interplay between battery constraints (power capacity, energy capacity, and minimum state-of-charge) and market prices for both "up" (discharge) and "down" (charge) directions.


---


## Table of Contents 
 
1. [Overview](https://chatgpt.com/?model=o1&temporary-chat=true#overview)
 
2. [Features](https://chatgpt.com/?model=o1&temporary-chat=true#features)
 
3. [Project Structure](https://chatgpt.com/?model=o1&temporary-chat=true#project-structure)
 
4. [Getting Started](https://chatgpt.com/?model=o1&temporary-chat=true#getting-started)  
  - [Prerequisites](https://chatgpt.com/?model=o1&temporary-chat=true#prerequisites)
 
  - [Installation](https://chatgpt.com/?model=o1&temporary-chat=true#installation)
 
5. [Usage](https://chatgpt.com/?model=o1&temporary-chat=true#usage)
 
6. [Visualization](https://chatgpt.com/?model=o1&temporary-chat=true#visualization)
 
7. [Contributing](https://chatgpt.com/?model=o1&temporary-chat=true#contributing)
 
8. [License](https://chatgpt.com/?model=o1&temporary-chat=true#license)
 
9. [Contact](https://chatgpt.com/?model=o1&temporary-chat=true#contact)


---


## Overview 

This project demonstrates:
 
- How to formulate and solve a deterministic optimization problem using **Pyomo** .

- How to handle both power and energy constraints for multiple battery assets.

- How to allocate capacity to "up" and "down" markets to maximize revenue.

- How to visualize both the solution (power bids over time) and the state of charge (SOC) of the batteries.

### Core Concept 

Each market can either be an "up" market (discharge) or a "down" market (charge). We have:
 
- A *fleet* of $$n_\text{assets}$$ battery systems.
 
- A set of $$n_\text{markets}$$ markets, each labeled with a direction `up` or `down`.
 
- $$n_\text{timesteps}$$ discrete time periods (e.g., hours).
 
- A **decision variable**  $$x_{t,m}$$ for every (time, market) pair (exactly $$n_\text{timesteps} \times n_\text{markets}$$ variables).

- An objective function that maximizes total revenue across all time steps and markets, taking into account each market’s price and the battery constraints.


---


## Features 
 
1. **Linear Optimization with Pyomo** 
Uses Pyomo’s `ConcreteModel` to define variables, objective, and constraints.
 
2. **Power and Energy Constraints** 
  - Each battery has a maximum charge/discharge power capacity per time step.

  - State-of-charge (SOC) tracking ensures the battery never exceeds its energy capacity or discharges below a minimum SOC threshold.
 
3. **Prequalification Limits** 
Each asset has a prequalified maximum power for each market, which sets an upper bound on $$x_{t,m}$$.
 
4. **Single-Shot Big-M Approach** 
  - Uses binary flags to indicate whether an asset is discharging or charging in a given hour (no simultaneous charge and discharge).
 
5. **Result Visualization**  
  - **State of Charge**  evolution over time for each asset.
 
  - **Bid Allocation**  over time for each market.


---


## Project Structure 


```bash
.
├── README.md               # This README file
├── requirements.txt        # Python dependencies (Pyomo, pandas, NumPy, etc.)
└── fleet_battery_opt.py    # Main script containing the optimization model
```
You can rename or split `fleet_battery_opt.py` into multiple files as needed (e.g., `model.py`, `visualization.py`, etc.).

---


## Getting Started 

### Prerequisites 

- Python 3.7+
 
- [GLPK]()  or another LP/MILP solver compatible with Pyomo.

- A working environment capable of installing Python packages (e.g., venv, conda).

### Installation 
 
1. **Clone**  the repository:

```bash
git clone https://github.com/your-username/fleet-battery-optimization.git
cd fleet-battery-optimization
```
 
2. **Install dependencies** :

```bash
pip install -r requirements.txt
```

This should install the main libraries:
 
  - `numpy`
 
  - `pandas`
 
  - `pyomo`
 
  - `plotly`
 
  - `matplotlib` (optional, if you use it)
 
3. **Ensure a solver**  (like `glpk`) is installed: 
  - For example, on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install glpk-utils
```
 
  - Or install via conda:

```bash
conda install -c conda-forge glpk
```


---


## Usage 
 
1. **Run the optimization** :

```bash
python fleet_battery_opt.py
```

This script will:

  - Generate random markets, directions, prequalified power, and prices.

  - Build and solve the optimization model using Pyomo.

  - Print solution details (objective value, chosen bids, and battery SOC).

  - Produce interactive Plotly figures showing SOC evolution and bids over time.
 
2. **Check the console output** : 
  - You will see messages about the solver status, the objective value, and any non-zero decisions $$x_{t,m}$$.

  - Battery SOC at each time step is also printed.
 
3. **View the plots** :
  - After the script finishes, two Plotly figures open in your default browser or in your IDE’s interactive window.
 
  - **SOC Evolution Over Time** 
Shows how each battery’s SOC changes from the initial state to the final time step.
 
  - **Bid Allocation Over Time** 
Shows how many MW are allocated to each market at each time step.


---


## Visualization 

By default, the code generates two Plotly figures:
 
1. **SOC Evolution Over Time** 
Illustrates how each battery’s energy content changes.
 
2. **Bid Allocation Over Time** 
Displays how much power is allocated to each "up" or "down" market in each time period.
