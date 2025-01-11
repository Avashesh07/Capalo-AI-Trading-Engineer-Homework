import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals, Objective, maximize, Constraint,
    Binary, ConstraintList, SolverFactory
)
from datetime import datetime

def main():
    # -----------------------------
    # Streamlit App Title
    # -----------------------------
    st.title("Single-Shot Big-M Approach with Pyomo")
    st.markdown(
        """
        This Streamlit app builds and solves a small energy-storage bidding model using Pyomo.
        Use the sidebar to set your parameters, then click **Solve**.
        """
    )

    # -----------------------------
    # Sidebar inputs
    # -----------------------------
    st.sidebar.header("Model Parameters")
    n_assets = st.sidebar.number_input(
        "Number of assets", min_value=1, value=2, step=1
    )
    n_markets = st.sidebar.number_input(
        "Number of markets", min_value=1, value=4, step=1
    )
    n_timesteps = st.sidebar.number_input(
        "Number of timesteps", min_value=1, value=6, step=1
    )
    asset_power_capacity = st.sidebar.number_input(
        "Asset power capacity (MW)", min_value=1, value=10, step=1
    )
    asset_energy_capacity = st.sidebar.number_input(
        "Asset energy capacity (MWh)", min_value=1, value=20, step=1
    )

    # Solve button
    if st.sidebar.button("Solve"):
        solve_and_display(
            n_assets, n_markets, n_timesteps,
            asset_power_capacity, asset_energy_capacity
        )

def solve_and_display(n_assets, n_markets, n_timesteps,
                      asset_power_capacity, asset_energy_capacity):
    """
    Builds and solves the Pyomo model, then displays results in Streamlit.
    """

    # For reproducibility
    np.random.seed(0)

    # Generate random directions for the markets
    directions = np.random.choice(["up", "down"], size=n_markets)
    assets = [f"a{i}" for i in range(n_assets)]
    markets = [f"m{i}_{direction}" for i, direction in enumerate(directions)]
    timesteps = pd.date_range(
        start="2025-01-01 00:00+00:00", periods=n_timesteps, freq="1H"
    )

    # Prequalified powers
    prequalified_powers = np.random.randint(
        0, asset_power_capacity, size=(len(assets), len(markets))
    )
    prequalified_powers = pd.DataFrame(
        columns=markets,
        index=assets,
        data=prequalified_powers
    )
    prequalified_powers.index.name = "asset_id"
    prequalified_powers.columns.name = "market"

    # Market prices
    market_prices = np.random.rand(len(timesteps), len(markets)).round(4) * 100
    market_prices = pd.DataFrame(
        columns=markets,
        index=timesteps,
        data=market_prices
    )
    market_prices.index.name = "datetime"
    market_prices.columns.name = "market"

    # direction_sign
    direction_sign = {
        m: 1 if "_up" in m else -1
        for m in markets
    }

    # -------------------------------
    #  CREATE MODEL
    # -------------------------------
    model = ConcreteModel("FleetWithExactBidVars")

    # 1) Sets
    model.TIME = Set(initialize=list(timesteps))
    model.MARKETS = Set(initialize=markets)
    model.ASSETS = Set(initialize=assets)

    # 2) Fleet-level decision variables: x[t,m]
    model.x = Var(model.TIME, model.MARKETS, domain=NonNegativeReals)

    # 3) SOC, up_aux, down_aux, and binary usage flags
    model.SOC = Var(
        model.ASSETS, model.TIME, domain=NonNegativeReals,
        bounds=(0, asset_energy_capacity)
    )
    model.use_discharge = Var(model.ASSETS, model.TIME, domain=Binary)
    model.use_charge = Var(model.ASSETS, model.TIME, domain=Binary)
    model.up_aux = Var(model.ASSETS, model.TIME, domain=NonNegativeReals)
    model.down_aux = Var(model.ASSETS, model.TIME, domain=NonNegativeReals)

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
    BIGM = asset_energy_capacity

    # (A) The total x[t,m] cannot exceed the sum of prequalified powers
    def fleet_prequalification_rule(m, tt, mm):
        sum_of_prequal = sum(prequalified_powers.loc[a, mm] for a in assets)
        return m.x[tt, mm] <= sum_of_prequal
    model.FleetPrequalConstraint = Constraint(
        model.TIME, model.MARKETS, rule=fleet_prequalification_rule
    )

    # (B) Each asset i must be able to handle the entire up or down portion
    def asset_discharge_rule(m, i, tt):
        up_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == 1)
        return up_sum <= asset_power_capacity
    model.AssetDischargeConstraint = Constraint(
        model.ASSETS, model.TIME, rule=asset_discharge_rule
    )

    def asset_charge_rule(m, i, tt):
        down_sum = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == -1)
        return down_sum <= asset_power_capacity
    model.AssetChargeConstraint = Constraint(
        model.ASSETS, model.TIME, rule=asset_charge_rule
    )

    # (C) Link up_aux/down_aux to usage flags
    model.UpAuxBounds = ConstraintList()
    for i in assets:
        for tt in model.TIME:
            model.UpAuxBounds.add(model.up_aux[i, tt] <= model.SOC[i, tt])
            model.UpAuxBounds.add(model.up_aux[i, tt] <= BIGM * model.use_discharge[i, tt])

    model.DownAuxBounds = ConstraintList()
    for i in assets:
        for tt in model.TIME:
            model.DownAuxBounds.add(model.down_aux[i, tt] <= asset_energy_capacity - model.SOC[i, tt])
            model.DownAuxBounds.add(model.down_aux[i, tt] <= BIGM * model.use_charge[i, tt])

    # (D) Sum of up_aux and down_aux across assets must match total up or down
    def up_sum_rule(m, tt):
        up_sum_ = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == 1)
        return up_sum_ == sum(m.up_aux[i, tt] for i in assets)
    model.UpSumConstraint = Constraint(model.TIME, rule=up_sum_rule)

    def down_sum_rule(m, tt):
        down_sum_ = sum(m.x[tt, mm] for mm in m.MARKETS if direction_sign[mm] == -1)
        return down_sum_ == sum(m.down_aux[i, tt] for i in assets)
    model.DownSumConstraint = Constraint(model.TIME, rule=down_sum_rule)

    # (E) No simultaneous charge/discharge
    model.NoSimultaneousChargeDischarge = ConstraintList()
    for i in assets:
        for tt in model.TIME:
            model.NoSimultaneousChargeDischarge.add(
                model.use_discharge[i, tt] + model.use_charge[i, tt] <= 1
            )

    # (F) SOC transitions
    model.SOCTransition = ConstraintList()
    for i in assets:
        for idx, t in enumerate(timesteps[:-1]):
            t_next = timesteps[idx + 1]
            model.SOCTransition.add(
                model.SOC[i, t_next] == model.SOC[i, t] + model.down_aux[i, t] - model.up_aux[i, t]
            )

    # (G) Initial SOC = 50%
    def initial_soc_rule(m, i):
        return m.SOC[i, timesteps[0]] == asset_energy_capacity * 0.5
    model.InitSOC = Constraint(model.ASSETS, rule=initial_soc_rule)

    # (H) Minimum SOC reserve
    MIN_SOC_PERCENTAGE = 0.1
    MIN_SOC = asset_energy_capacity * MIN_SOC_PERCENTAGE

    def prevent_discharge_below_min_soc_rule(m, i):
        final_timestep = timesteps[-1]
        return m.up_aux[i, final_timestep] <= m.SOC[i, final_timestep] - MIN_SOC
    model.PreventDischargeBelowMinSOC = Constraint(
        model.ASSETS, rule=prevent_discharge_below_min_soc_rule
    )

    # -------------------------------
    #   MATHEMATICAL FORMULATION
    # -------------------------------
    st.subheader("Mathematical Formulation")
    st.markdown(r"""
\[
\textbf{Decision Variables:} 
\begin{aligned}
& x_{t,m} \ge 0  && \quad \forall \, t \in T, \, m \in M, \\
& \text{SOC}_{i,t} \ge 0 && \quad \forall \, i \in A, \, t \in T, \\
& \text{use\_discharge}_{i,t} \in \{0, 1\} && \quad \forall \, i \in A, \, t \in T, \\
& \text{use\_charge}_{i,t} \in \{0, 1\} && \quad \forall \, i \in A, \, t \in T, \\
& \text{up\_aux}_{i,t} \ge 0 && \quad \forall \, i \in A, \, t \in T, \\
& \text{down\_aux}_{i,t} \ge 0 && \quad \forall \, i \in A, \, t \in T. \\
\end{aligned}
\]

\[
\textbf{Objective:} \quad 
\max \sum_{t \in T} \sum_{m \in M} \bigl[\text{dir}(m) \cdot p_{t,m} \bigr] \cdot x_{t,m}
\]
where \(\text{dir}(m)\in \{+1, -1\}\) (depending on whether the market is up or down),
and \(p_{t,m}\) is the price for market \(m\) at time \(t\).

\[
\textbf{Constraints:}
\]
- **(A) Fleet Prequalification**  
  \[
    x_{t,m} \;\le\; \sum_{i \in A} \text{PrequalPower}_{i,m}
    \quad \forall t,m.
  \]

- **(B) Per-Asset Power Limit**  
  \[
    \sum_{m \in M : \text{dir}(m)=+1} x_{t,m} \;\le\; \text{PowerCapacity}, 
    \quad
    \sum_{m \in M : \text{dir}(m)=-1} x_{t,m} \;\le\; \text{PowerCapacity}
    \quad \forall i,t.
  \]

- **(C) up\_aux and down\_aux Bounds**  
  \[
    \begin{aligned}
    &\text{up\_aux}_{i,t} \le \text{SOC}_{i,t}, && 
        \text{up\_aux}_{i,t} \le \text{BIGM} \cdot \text{use\_discharge}_{i,t}, \\
    &\text{down\_aux}_{i,t} \le (\text{Capacity} - \text{SOC}_{i,t}), &&
        \text{down\_aux}_{i,t} \le \text{BIGM} \cdot \text{use\_charge}_{i,t}.
    \end{aligned}
  \]

- **(D) Summation of up\_aux and down\_aux**  
  \[
    \sum_{m : \text{dir}(m)=+1} x_{t,m} \;=\; \sum_{i} \text{up\_aux}_{i,t}, \qquad
    \sum_{m : \text{dir}(m)=-1} x_{t,m} \;=\; \sum_{i} \text{down\_aux}_{i,t}.
  \]

- **(E) No Simultaneous Charge and Discharge**  
  \[
    \text{use\_discharge}_{i,t} + \text{use\_charge}_{i,t} \le 1 
    \quad \forall i,t.
  \]

- **(F) SOC Transition**  
  \[
    \text{SOC}_{i,t+1} = \text{SOC}_{i,t} + \text{down\_aux}_{i,t} - \text{up\_aux}_{i,t}.
  \]

- **(G) Initial SOC**  
  \[
    \text{SOC}_{i,t_0} = 0.5 \cdot \text{Capacity}.
  \]

- **(H) Minimum SOC at Final Timestep**  
  \[
    \text{up\_aux}_{i,T_{\text{final}}} \le \text{SOC}_{i,T_{\text{final}}} - \text{minSOC}.
  \]

    """)

    # -------------------------------
    #   Check number of x vars
    # -------------------------------
    num_x_vars = len(model.x)
    expected_x_vars = n_timesteps * n_markets

    # Display the decision variable count check
    st.subheader("Decision Variable Count Check")
    st.write(f"Number of `x[t,m]` variables in `model.x`: **{num_x_vars}**")
    st.write(f"Expected number of `x[t,m]` variables: **{expected_x_vars}**")

    if num_x_vars == expected_x_vars:
        st.success("The number of x[t,m] variables matches n_timesteps * n_markets.")
    else:
        st.warning("WARNING: The number of x[t,m] variables does NOT match n_timesteps * n_markets.")

    # -------------------------------
    #   Solve
    # -------------------------------
    solver = SolverFactory("glpk")
    res = solver.solve(model, tee=False, keepfiles=False)

    # Retrieve objective value
    objective_value = model.Obj()

    st.success(f"Optimal Objective: {objective_value:.2f} EUR")

    # Display nonzero x[t, m] results
    st.subheader("Bid Variable Results (x[t,m])")
    x_results = []
    for tt in model.TIME:
        for mm in model.MARKETS:
            val = model.x[tt, mm].value
            if abs(val) > 1e-6:
                x_results.append(
                    {"Datetime": tt, "Market": mm, "Value (MW)": round(val, 3)}
                )
    if x_results:
        st.dataframe(pd.DataFrame(x_results))
    else:
        st.write("All bids are 0.")

    # Display final state of charge per asset
    st.subheader("Final State of Charge per Asset (MWh)")
    soc_summary = []
    for i in model.ASSETS:
        # SOC at final node
        last_t = timesteps[-1]
        final_soc = (
            model.SOC[i, last_t].value
            - model.up_aux[i, last_t].value
            + model.down_aux[i, last_t].value
        )
        soc_summary.append({"Asset": i, "Final SOC (MWh)": round(final_soc, 3)})
    st.table(pd.DataFrame(soc_summary))

    # -------------------------------
    # Create figures using Plotly
    # -------------------------------

    # 1) SOC Evolution Over Time
    soc_fig = go.Figure()
    for i in assets:
        soc_values = [model.SOC[i, tt].value for tt in timesteps]
        soc_fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=soc_values,
                mode='lines+markers',
                name=f"SOC {i}",
                hovertemplate='Time: %{x}<br>SOC: %{y:.2f} MWh',
                line=dict(width=3),
                marker=dict(size=8)
            )
        )

    soc_fig.update_layout(
        title=dict(
            text="SOC Evolution Over Time",
            font=dict(size=30, color="darkblue"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=16)),
            tickfont=dict(size=12, color="gray"),
            tickformat="%H:%M",
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title=dict(text="SOC (MWh)", font=dict(size=16)),
            tickfont=dict(size=12, color="gray"),
            showgrid=True,
            gridcolor="lightgray",
        ),
        legend=dict(
            title=dict(text="Assets", font=dict(size=14)),
            font=dict(size=12, color="gray"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(soc_fig, use_container_width=True)

    # 2) Bid Allocation Over Time
    bid_fig = go.Figure()
    for mm in markets:
        mm_vals = [model.x[tt, mm].value for tt in timesteps]
        bid_fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=mm_vals,
                mode='lines+markers',
                name=f"Bid {mm}",
                hovertemplate='Time: %{x}<br>Bid: %{y:.2f} MW',
                line=dict(width=3),
                marker=dict(size=8)
            )
        )

    bid_fig.update_layout(
        title=dict(
            text="Bid Allocation Over Time",
            font=dict(size=30, color="darkblue"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=16)),
            tickfont=dict(size=12, color="gray"),
            tickformat="%H:%M",
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title=dict(text="Bid (MW)", font=dict(size=16)),
            tickfont=dict(size=12, color="gray"),
            showgrid=True,
            gridcolor="lightgray",
        ),
        legend=dict(
            title=dict(text="Markets", font=dict(size=14)),
            font=dict(size=12, color="gray"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(bid_fig, use_container_width=True)

    st.markdown("#### Single-Shot Big-M Approach Done!")

if __name__ == "__main__":
    main()
