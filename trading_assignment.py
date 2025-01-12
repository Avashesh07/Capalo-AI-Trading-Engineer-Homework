import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyomo.environ import *

np.random.seed(0)

n_assets             = 2
n_markets            = 4
n_timesteps          = 6
asset_power_capacity = 10  # max charging or discharging power (MW) per asset
asset_energy_capacity = 20  # max energy capacity (MWh) per asset

# Generate random directions for the markets
directions = np.random.choice(["up", "down"], size = n_markets)

# Create labels
assets     = [f"a{i}" for i in range(n_assets)]
markets    = [f"m{i}_{direction}" for i, direction in enumerate(directions)]
timesteps  = pd.date_range(start = "2025-01-01 00:00+00:00", 
                           periods = n_timesteps, 
                           freq = "1h")

# Prequalified power (MW) for each asset-market
prequalified_powers = np.random.randint(0, asset_power_capacity, 
                                        size = (len(assets), len(markets)))
prequalified_powers = pd.DataFrame( 
    columns = markets,
    index   = assets,
    data    = prequalified_powers
)
prequalified_powers.index.name = "asset_id" 
prequalified_powers.columns.name = "market"

# Market prices (EUR/MW) for each market and time
market_prices = np.random.rand(len(timesteps), len(markets)).round(4) * 100

market_prices = pd.DataFrame(
    columns = markets,
    index   = timesteps,
    data    = market_prices
)
market_prices.index.name   = "datetime"
market_prices.columns.name = "market"

# Print directions
print("\n--- Market Directions ---")
print(f"Directions chosen (up or down) for each market: {', '.join(directions)}")

# Print markets
print("\n--- Markets ---")
for i, market in enumerate(markets):
    print(f"Market {i + 1}: {market}")

# Print assets
print("\n--- Assets ---")
for i, asset in enumerate(assets):
    print(f"Asset {i + 1}: {asset}")

# Print timesteps
formatted_timesteps = timesteps.strftime('%Y-%m-%d %H:%M')
print("\n--- Timesteps ---")
for i, t in enumerate(formatted_timesteps):
    print(f"Timestep {i + 1}: {t}")

# Print prequalified powers
print("\n--- Prequalified Powers (MW) ---")
print(prequalified_powers.to_string(index=True))

# Print market prices
print("\n--- Market Prices (EUR/MW) ---")
print(market_prices.to_string(index=True))

# +1 if market is 'up' (discharge), -1 if 'down' (charge)
direction_sign = {}
for m in markets:
    if "_up" in m:
        direction_sign[m] = +1
    else:
        direction_sign[m] = -1

print("Direction sign dictionary:", direction_sign, "\n")

# -------------------------------
#  CREATE MODEL
# -------------------------------
model = ConcreteModel("FleetWithExactBidVars")


# 1) Sets
model.TIME    = Set(initialize=list(timesteps))
model.MARKETS = Set(initialize=markets)
model.ASSETS  = Set(initialize=assets)

# Double-check set lengths
print(f"Number of timesteps: {len(model.TIME)} (expected {n_timesteps})")
print(f"Number of markets:   {len(model.MARKETS)} (expected {n_markets})")
print(f"Number of assets:    {len(model.ASSETS)} (expected {n_assets})\n")

# 2) Fleet-level decision variables: x[t,m]
#    EXACTLY n_timesteps * n_markets variables
model.x = Var(model.TIME, model.MARKETS, domain=NonNegativeReals)

# Print out how many x variables we have:
num_x_vars = len(model.x)
print(f"Number of x variables in 'model.x' = {num_x_vars}")
print(f"Expected number of x variables = {n_timesteps * n_markets}")
if num_x_vars == n_timesteps * n_markets:
    print("Great! The number of x variables matches n_timesteps * n_markets.\n")
else:
    print("WARNING: The number of x variables does NOT match n_timesteps * n_markets.\n")

# 3) We STILL need each asset's SOC as an auxiliary variable
model.SOC = Var(model.ASSETS, model.TIME, domain=NonNegativeReals, 
                bounds=(0, asset_energy_capacity))

# Some big-M approach:
# We'll introduce a small set of binary "discharge feasible" flags for each asset-time
# that effectively say "Asset i is used for discharge at time t" up to some fraction.

model.use_discharge = Var(model.ASSETS, model.TIME, domain=Binary)
model.use_charge    = Var(model.ASSETS, model.TIME, domain=Binary)

# 4) Objective: Maximize total revenue
def obj_rule(m):
    return sum(
        direction_sign[mm] * market_prices.loc[tt, mm] * m.x[tt, mm]
        for tt in m.TIME
        for mm in m.MARKETS
    )
model.Obj = Objective(rule=obj_rule, sense=maximize)

# ---------------------------------------------------
#   CONSTRAINTS
# ---------------------------------------------------

# (A) The total x[t,m] cannot exceed the sum of prequalified powers across assets
def fleet_prequalification_rule(m, tt, mm):
    sum_of_prequal = sum(prequalified_powers.loc[a, mm] for a in assets)
    return m.x[tt, mm] <= sum_of_prequal
model.FleetPrequalConstraint = Constraint(model.TIME, model.MARKETS, 
                                          rule=fleet_prequalification_rule)

# (B) Each asset i must be able to handle the entire up or down portion
def asset_discharge_rule(m, i, tt):
    up_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == +1)
    return up_sum <= asset_power_capacity
model.AssetDischargeConstraint = Constraint(model.ASSETS, model.TIME, 
                                            rule=asset_discharge_rule)

def asset_charge_rule(m, i, tt):
    down_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == -1)
    return down_sum <= asset_power_capacity
model.AssetChargeConstraint = Constraint(model.ASSETS, model.TIME, 
                                         rule=asset_charge_rule)

# (C) SoC constraints
BIGM = asset_energy_capacity  # a big enough constant for "worst case"

model.up_aux = Var(model.ASSETS, model.TIME, domain=NonNegativeReals)   # how much each asset can discharge
model.down_aux = Var(model.ASSETS, model.TIME, domain=NonNegativeReals) # how much each asset can charge

# Link up_aux to the binary "use_discharge"
def up_aux_bounds_rule(m, i, tt):
    # up_aux[i,t] <= SOC[i,t]
    # up_aux[i,t] <= BIGM * use_discharge[i,t]
    return [
        m.up_aux[i,tt] <= m.SOC[i,tt],
        m.up_aux[i,tt] <= BIGM * m.use_discharge[i,tt]
    ]
model.UpAuxBounds = ConstraintList()
for i in assets:
    for tt in model.TIME:
        bnds = up_aux_bounds_rule(model, i, tt)
        for con in bnds:
            model.UpAuxBounds.add(con)

# Similarly for down_aux
def down_aux_bounds_rule(m, i, tt):
    # can't exceed (capacity - SOC)
    # can't exceed BIGM * use_charge[i,t]
    return [
        m.down_aux[i,tt] <= asset_energy_capacity - m.SOC[i,tt],
        m.down_aux[i,tt] <= BIGM * m.use_charge[i,tt]
    ]
model.DownAuxBounds = ConstraintList()
for i in assets:
    for tt in model.TIME:
        bnds = down_aux_bounds_rule(model, i, tt)
        for con in bnds:
            model.DownAuxBounds.add(con)

# Now, the sum of up_aux across all assets must match the total up_sum.
# same for down_aux
def up_sum_rule(m, tt):
    up_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == +1)
    return up_sum == sum(m.up_aux[i,tt] for i in assets)
model.UpSumConstraint = Constraint(model.TIME, rule=up_sum_rule)

def down_sum_rule(m, tt):
    down_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == -1)
    return down_sum == sum(m.down_aux[i,tt] for i in assets)
model.DownSumConstraint = Constraint(model.TIME, rule=down_sum_rule)

model.NoSimultaneousChargeDischarge = ConstraintList()
for i in assets:
    for tt in model.TIME:
        model.NoSimultaneousChargeDischarge.add(
            model.use_discharge[i, tt] + model.use_charge[i, tt] <= 1
        )

# Finally, real SOC transitions:
# SOC[i, t+1] = SOC[i, t] + down_aux[i,t] - up_aux[i,t]
# no more "worst-case bounding" needed
def soc_transition_rule(m, i, t):
    # skip last
    if t == timesteps[-1]:
        return Constraint.Skip
    idx = list(timesteps).index(t)
    t_next = timesteps[idx+1]
    return m.SOC[i, t_next] == m.SOC[i,t] + m.down_aux[i,t] - m.up_aux[i,t]
model.SOCTransition = ConstraintList()
for i in assets:
    for idx, t in enumerate(timesteps[:-1]):
        model.SOCTransition.add(soc_transition_rule(model, i, t))

def initial_soc_rule(m, i):
    return m.SOC[i, timesteps[0]] == asset_energy_capacity * 0.5  # Start at 50% capacity
model.InitSOC = Constraint(model.ASSETS, rule=initial_soc_rule)

MIN_SOC_PERCENTAGE = 0.1  # Reserve 10% of the battery capacity
MIN_SOC = asset_energy_capacity * MIN_SOC_PERCENTAGE

# Constraint to prevent discharge below the minimum SOC at the final timestep
def prevent_discharge_below_min_soc_rule(m, i):
    final_timestep = timesteps[-1]
    return m.up_aux[i, final_timestep] <= m.SOC[i, final_timestep] - MIN_SOC
model.PreventDischargeBelowMinSOC = Constraint(model.ASSETS, rule=prevent_discharge_below_min_soc_rule)


# Print how many constraints we have in total
all_constraints = list(model.component_objects(Constraint, active=True))
print(f"Total number of active constraints = {sum(1 for _ in all_constraints)}\n")



# -------------------------------
# SOLVE
# -------------------------------
solver = SolverFactory("glpk")
res = solver.solve(model, tee=True, keepfiles=True)
print("LP File Path:", res.solver.status)

print(f"OPT OBJ: {model.Obj()} EUR")

for tt in model.TIME:
    for mm in model.MARKETS:
        val = model.x[tt, mm].value
        if abs(val) > 1e-6:
            print(f"x[{tt},{mm}] = {val:.3f} MW")

for i in model.ASSETS:
    for tt in model.TIME:
        print(f"SOC[{i},{tt}] = {model.SOC[i,tt].value:.3f}")
    # Print the final SOC explicitly for each asset
    last_time = timesteps[-1]
    final_soc = (
        model.SOC[i, last_time].value
        - model.up_aux[i, last_time].value
        + model.down_aux[i, last_time].value
    )

    print(f"\n--- Final SOC after last bid for {i} = {final_soc:.3f} ---\n")



# SOC Evolution Over Time
soc_fig = go.Figure()

for i in assets:
    soc_fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=[model.SOC[i, tt].value for tt in timesteps],
            mode='lines+markers',
            name=f"SOC {i}",
            hovertemplate='Time: %{x}<br>SOC: %{y:.2f} MWh',
            line=dict(width=3),  # Line width
            marker=dict(size=8)  # Marker size
        )
    )

# Layout customization
soc_fig.update_layout(
    title=dict(
        text="SOC Evolution Over Time",
        font=dict(family="Montserrat, sans-serif", size=30, color="darkblue"),
        x=0.5  # Center align title
    ),
    xaxis=dict(
        title=dict(
            text="Time",
            font=dict(family="Montserrat, sans-serif", size=16, color="black")
        ),
        tickfont=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        tickformat="%H:%M",
        showgrid=True,
        gridcolor="lightgray"
    ),
    yaxis=dict(
        title=dict(
            text="SOC (MWh)",
            font=dict(family="Montserrat, sans-serif", size=16, color="black")
        ),
        tickfont=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        showgrid=True,
        gridcolor="lightgray"
    ),
    legend=dict(
        title=dict(
            text="Assets",
            font=dict(family="Montserrat, sans-serif", size=14, color="black")
        ),
        font=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template="plotly_white",
    hovermode="x unified",
    font=dict(family="Montserrat, sans-serif", color="black")  # General font for the graph
)

soc_fig.show()

# Bid Allocation Over Time
bid_fig = go.Figure()

for mm in markets:
    bid_fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=[model.x[tt, mm].value for tt in timesteps],
            mode='lines+markers',
            name=f"Bid {mm}",
            hovertemplate='Time: %{x}<br>Bid: %{y:.2f} MW',
            line=dict(width=3),  # Line width
            marker=dict(size=8)  # Marker size
        )
    )

# Layout customization
bid_fig.update_layout(
    title=dict(
        text="Bid Allocation Over Time",
        font=dict(family="Montserrat, sans-serif", size=30, color="darkblue"),
        x=0.5
    ),
    xaxis=dict(
        title=dict(
            text="Time",
            font=dict(family="Montserrat, sans-serif", size=16, color="black")
        ),
        tickfont=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        tickformat="%H:%M",
        showgrid=True,
        gridcolor="lightgray"
    ),
    yaxis=dict(
        title=dict(
            text="Bid (MW)",
            font=dict(family="Montserrat, sans-serif", size=16, color="black")
        ),
        tickfont=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        showgrid=True,
        gridcolor="lightgray"
    ),
    legend=dict(
        title=dict(
            text="Markets",
            font=dict(family="Montserrat, sans-serif", size=14, color="black")
        ),
        font=dict(family="Montserrat, sans-serif", size=12, color="gray"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template="plotly_white",
    hovermode="x unified",
    font=dict(family="Montserrat, sans-serif", color="black")
)

bid_fig.show()


print("\n--- Single-Shot Big-M Approach Done ---\n")



