#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG: Analyze power sources for Aug 9-10 in Design 1
Q: If battery discharge = 0 kWh, where is the served load coming from?
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# Load the simulation results from run_simulation.py (assuming it was already run)
print("=" * 140)
print("【ANALYSIS: Aug 9-10 Power Balance for Design 1】")
print("=" * 140)

# Load weather data to get timestamps
df_weather = pd.read_csv('Morakot_weather_with_demand.csv')
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

# Find Aug 9-10 indices
idx_list = []
for t in range(len(df_weather)):
    ts_str = str(df_weather['datetime'].iloc[t])
    if '2009-08-09' in ts_str or '2009-08-10' in ts_str:
        idx_list.append(t)

if not idx_list:
    print("ERROR: Aug 9-10 data not found!")
    sys.exit(1)

idx_start, idx_end = idx_list[0], idx_list[-1]

print(f"\nTime range: {df_weather['datetime'].iloc[idx_start]} to {df_weather['datetime'].iloc[idx_end]}")
print(f"Time indices: t={idx_start} to t={idx_end} ({idx_end - idx_start + 1} hours)\n")

# Now we need to reconstruct what the simulation found
# The key question: WT + PV + DG + Battery_discharge = Served_load?
# If battery_discharge ≈ 0, then WT + PV + DG should ≈ Served_load

# Let's look at the stored CSV files from the latest simulation run
import glob

# Check if there's a CSV file with Design 1 results
csv_files = glob.glob("charts/design_1/*.csv")
print(f"Found CSV files in charts/design_1/: {csv_files}\n")

if csv_files:
    # Try to read the first CSV
    try:
        df_results = pd.read_csv(csv_files[0])
        print(f"Loaded {csv_files[0]}")
        print(f"Columns: {list(df_results.columns)}\n")
        
        # Filter for Aug 9-10
        df_results['datetime'] = pd.to_datetime(df_results.get('datetime', df_results.get('Time', None)))
        df_aug = df_results[(df_results['datetime'].dt.date >= pd.to_datetime('2009-08-09').date()) & 
                             (df_results['datetime'].dt.date <= pd.to_datetime('2009-08-10').date())]
        
        if len(df_aug) > 0:
            print(f"Found {len(df_aug)} hours of Aug 9-10 data in CSV\n")
            print(df_aug[['datetime', 'P_WT', 'P_PV', 'P_DG', 'P_discharge', 'Demand', 'Served', 'Unserved']].head(20))
        else:
            print("CSV does not contain Aug 9-10 data\n")
    except Exception as e:
        print(f"Error loading CSV: {e}\n")

# Since CSV might not be available, let's re-run the simulation just for Design 1
print("\n" + "="*140)
print("RUNNING SIMULATION FOR DESIGN 1 TO GET CURRENT DATA")
print("="*140 + "\n")

from microgrid.models import (
    MicrogridDesign,
    CostParameters,
    EMSPolicy,
    DisturbanceScenario,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather

# Build Design 1 (same as in run_simulation.py)
P_WT_1 = [1000.0] * 2
P_PV_1 = [2500.0] * 20
P_DG_1 = [5000.0] * 1
P_BAT_1 = [4000.0] * 2

B_max_list = [200.0] * 8
B_init_list = [100.0] * 8

def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    return fuel_rate_max * 24 * days_fuel * (sum(P_DG) / 5000.0) * 1.333

design_1 = MicrogridDesign(
    P_WT=P_WT_1,
    P_PV=P_PV_1,
    P_DG=P_DG_1,
    P_BAT=P_BAT_1,
    B_max=B_max_list,
    B_init=B_init_list,
    eta_c=0.97,
    eta_d=0.98,
    C_WT=0.22,
    C_PV=0.36,
    A_WT=0.98,
    fuel_rate_max=50.0,
    fuel_storage=compute_fuel_storage(50.0, 20, P_DG_1),
    DG_min_loading=0.2,
    DG_max_loading=0.8,
    B_min_soc_frac=0.2,
    B_max_soc_frac=0.8,
    C_rate_charge=1.0,
    C_rate_discharge=1.0,
)

# Build time, hazard, and scenario
print("[1] Building time input and hazard...")
demand_series = df_weather["demand"].tolist()
time_input = build_time_input(df_weather, demand_series)
hazard = build_hazard_from_weather(df_weather)

disturbance_start_ts = pd.to_datetime("2009-08-07T00:00:00")
disturbance_end_ts = pd.to_datetime("2009-08-10T00:00:00")

# Find indices for disturbance times
try:
    disturbance_start_idx = df_weather.index[df_weather["datetime"] == disturbance_start_ts][0]
    disturbance_end_idx = df_weather.index[df_weather["datetime"] == disturbance_end_ts][0]
except IndexError:
    disturbance_start_idx = (df_weather["datetime"] - disturbance_start_ts).abs().idxmin()
    disturbance_end_idx = (df_weather["datetime"] - disturbance_end_ts).abs().idxmin()

scenario = DisturbanceScenario(
    name="Morakot_2009",
    disturbance_start=disturbance_start_idx,
    disturbance_end=disturbance_end_idx,
    base_p_damage_WT=0.10,
    base_p_damage_PV=0.05,
    base_p_damage_DG=0.01,
    base_p_damage_BAT=0.01,
    MTTR_WT=72.0,
    MTTR_PV=31.0,
    MTTR_DG=289.0,
    MTTR_BAT=3.0,
    grid_MTTR_hours=48,
    hazard=hazard,
    evaluation_horizon_hours=168,
    annual_occurrence_rate=0.1,
)

# Run simulation
print("\n[2] Running Design 1 simulation...")
sim_result = simulate_microgrid_resilience(
    design=design_1,
    scenario=scenario,
    time_input=time_input,
    ems_policy=EMSPolicy(dg_start_soc=0.30, dg_stop_soc=0.70),
)
print(f"    EENS={sim_result.EENS:.1f}, Fuel used={sim_result.fuel_used:.1f} L\n")

# Extract Aug 9-10 data
print("[3] Extracting Aug 9-10 data...\n")

print(f"{'Time':<20} {'WT':>10} {'PV':>10} {'DG':>10} {'Batt_Disch':>12} {'Total_Gen':>12} {'Demand':>10} {'Served':>10} {'Unserved':>10} {'SOC':>7} {'Mode':<20}")
print("-"*155)

total_wt = 0.0
total_pv = 0.0
total_dg = 0.0
total_discharge = 0.0
total_demand = 0.0
total_served = 0.0
total_unserved = 0.0
total_from_sources = 0.0

for t in range(idx_start, idx_end + 1):
    ts = df_weather['datetime'].iloc[t]
    
    p_wt = sim_result.P_wt[t]
    p_pv = sim_result.P_pv[t]
    p_dg = sim_result.P_dg[t]
    p_discharge = sum(sim_result.P_discharge[i][t] for i in range(len(sim_result.P_discharge)))
    
    demand = sim_result.demand[t]
    served = sim_result.Gt[t]
    unserved = sim_result.Tt[t]
    avg_soc = sim_result.avg_soc_frac[t]
    ems_mode = sim_result.ems_mode[t]
    
    total_gen = p_wt + p_pv + p_dg + p_discharge
    
    total_wt += p_wt
    total_pv += p_pv
    total_dg += p_dg
    total_discharge += p_discharge
    total_demand += demand
    total_served += served
    total_unserved += unserved
    total_from_sources += total_gen
    
    time_str = str(ts)[:16]
    print(f"{time_str:<20} {p_wt:>10.1f} {p_pv:>10.1f} {p_dg:>10.1f} {p_discharge:>12.1f} {total_gen:>12.1f} {demand:>10.1f} {served:>10.1f} {unserved:>10.1f} {avg_soc:>7.3f} {ems_mode:<20}")

print("-"*155)
print(f"\n【SUMMARY FOR AUG 9-10】")
print(f"Wind (WT) generation:        {total_wt:>10.1f} kWh")
print(f"Solar (PV) generation:       {total_pv:>10.1f} kWh")
print(f"Diesel (DG) generation:      {total_dg:>10.1f} kWh")
print(f"Battery discharge:           {total_discharge:>10.1f} kWh")
print(f"{'':>30}{'─'*15}")
print(f"TOTAL GENERATION:            {total_from_sources:>10.1f} kWh")
print(f"\nTotal demand:                {total_demand:>10.1f} kWh")
print(f"Total served:                {total_served:>10.1f} kWh")
print(f"Total unserved:              {total_unserved:>10.1f} kWh")

print(f"\n【ANALYSIS】")
print(f"Power balance: Generated ({total_from_sources:.1f}) vs Served ({total_served:.1f})")
if abs(total_from_sources - total_served) < 1.0:
    print("✓ Power balance MATCHES: All generation goes to load")
else:
    print(f"✗ Power balance MISMATCH: Difference = {total_from_sources - total_served:.1f} kWh")

if total_discharge == 0.0:
    print(f"\n✓ Battery discharge is ZERO (as expected)")
    print(f"  → All served load comes from: WT + PV + DG")
    print(f"  → WT contribution: {total_wt/total_served*100:.1f}%")
    print(f"  → PV contribution: {total_pv/total_served*100:.1f}%")
    print(f"  → DG contribution: {total_dg/total_served*100:.1f}%")
else:
    print(f"\n✗ Battery discharge is NOT zero: {total_discharge:.1f} kWh")
    print(f"  This contradicts the earlier finding!")

# Check if all values are really zero for wind/pv/dg
if total_wt == 0.0 and total_pv == 0.0:
    print(f"\n⚠️  WARNING: Both WT and PV are ZERO during Aug 9-10")
    print(f"   → All {total_served:.1f} kWh served load must come from DG: {total_dg:.1f} kWh")
    if total_dg == 0.0:
        print(f"   → But DG is also ZERO!")
        print(f"   → This suggests the served load of {total_served:.1f} kWh cannot be explained!")
