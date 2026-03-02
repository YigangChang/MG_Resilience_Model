#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG: Why does DG never activate during the disaster period?
Trace DG activation conditions during island mode
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from microgrid.models import (
    MicrogridDesign,
    EMSPolicy,
    DisturbanceScenario,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather

print("=" * 140)
print("【DEBUG: Why DG Never Activates During Disaster Period】")
print("=" * 140)

# Load weather data
df_weather = pd.read_csv('Morakot_weather_with_demand.csv')
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
print(f"\nLoaded {len(df_weather)} hours of weather data")

# Setup Design 1
P_WT_1 = [1000.0] * 2
P_PV_1 = [2500.0] * 20
P_DG_1 = [5000.0] * 1
P_BAT_1 = [4000.0] * 2

B_max_list = [200.0] * 8
B_init_list = [100.0] * 8

def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    n_dg = len(P_DG)
    return fuel_rate_max * 24 * days_fuel * n_dg

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

print("\n【DESIGN 1 PARAMETERS】")
print(f"  DG Capacity: {sum(P_DG_1)} kW ({len(P_DG_1)} units)")
print(f"  DG Min Loading: {design_1.DG_min_loading*100}% = {sum(P_DG_1)*design_1.DG_min_loading} kW")
print(f"  DG Max Loading: {design_1.DG_max_loading*100}% = {sum(P_DG_1)*design_1.DG_max_loading} kW")
print(f"  Fuel Storage: {design_1.fuel_storage} gal")
print(f"  Fuel Rate Max: {design_1.fuel_rate_max} gal/hr")
print(f"  Battery Total Capacity: {sum(B_max_list)} kWh")
print(f"  Battery Power Rating: {sum(P_BAT_1)} kW")

# Build time input and scenario
demand_series = df_weather["demand"].tolist()
time_input = build_time_input(df_weather, demand_series)
hazard = build_hazard_from_weather(df_weather)

disturbance_start_ts = pd.to_datetime("2009-08-07T00:00:00")
disturbance_end_ts = pd.to_datetime("2009-08-10T00:00:00")

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

ems_policy = EMSPolicy(dg_start_soc=0.30, dg_stop_soc=0.70)

print(f"\n【EMS POLICY】")
print(f"  DG Start SOC: {ems_policy.dg_start_soc*100:.0f}%")
print(f"  DG Stop SOC:  {ems_policy.dg_stop_soc*100:.0f}%")

# Run simulation
print(f"\n【RUNNING SIMULATION】")
sim_result = simulate_microgrid_resilience(
    design=design_1,
    scenario=scenario,
    time_input=time_input,
    ems_policy=ems_policy,
)

print(f"  EENS: {sim_result.EENS:.1f} kWh")
print(f"  Fuel Used: {sim_result.fuel_used:.1f} L")
print(f"  Total DG Generation: {sum(sim_result.P_dg):.1f} kWh")

# Find disaster period (island mode)
island_periods = []
for t in range(len(sim_result.ems_mode)):
    if 'island' in sim_result.ems_mode[t].lower():
        island_periods.append(t)

if island_periods:
    island_start = island_periods[0]
    island_end = island_periods[-1]
    
    print(f"\n【ISLAND MODE PERIOD】")
    print(f"  Start: {df_weather['datetime'].iloc[island_start]}")
    print(f"  End:   {df_weather['datetime'].iloc[island_end]}")
    print(f"  Duration: {island_end - island_start + 1} hours")
    
    # Analyze DG activation conditions during island mode
    print(f"\n{'Time':<20} {'SOC':>7} {'WT':>8} {'PV':>8} {'Batt':>8} {'Demand':>10} {'Deficit':>10} {'DG':>8} {'Mode':<20}")
    print("-" * 130)
    
    dg_eligible_hours = 0
    low_soc_hours = 0
    deficit_hours = 0
    
    for t in island_periods[:50]:  # Show first 50 hours
        ts = df_weather['datetime'].iloc[t]
        soc = sim_result.avg_soc_frac[t]
        p_wt = sim_result.P_wt[t]
        p_pv = sim_result.P_pv[t]
        p_batt = sum(sim_result.P_discharge[i][t] for i in range(len(sim_result.P_discharge)))
        demand = sim_result.demand[t]
        p_dg = sim_result.P_dg[t]
        mode = sim_result.ems_mode[t]
        
        renewable_total = p_wt + p_pv
        deficit = demand - renewable_total - p_batt
        
        if soc <= ems_policy.dg_start_soc:
            low_soc_hours += 1
        if deficit > 0:
            deficit_hours += 1
        if soc <= ems_policy.dg_start_soc and deficit > 0:
            dg_eligible_hours += 1
        
        time_str = str(ts)[:16]
        marker = ""
        if soc <= ems_policy.dg_start_soc and deficit > 0 and p_dg == 0:
            marker = " ⚠️ DG SHOULD ACTIVATE"
        
        print(f"{time_str:<20} {soc:>7.3f} {p_wt:>8.0f} {p_pv:>8.0f} {p_batt:>8.0f} {demand:>10.0f} {deficit:>10.0f} {p_dg:>8.0f} {mode:<20}{marker}")
    
    print("-" * 130)
    print(f"\n【ANALYSIS】")
    print(f"  Total island hours analyzed: {len(island_periods[:50])}")
    print(f"  Hours with SOC ≤ {ems_policy.dg_start_soc*100:.0f}%: {low_soc_hours}")
    print(f"  Hours with deficit > 0: {deficit_hours}")
    print(f"  Hours eligible for DG (SOC low + deficit): {dg_eligible_hours}")
    print(f"  Actual DG activations: {sum(1 for t in island_periods if sim_result.P_dg[t] > 0)}")
    
    # Check if DG units are available
    dg_failures = 0
    print(f"\n【DG UNIT STATUS】")
    for i in range(len(P_DG_1)):
        # Check if DG failed during island period
        if hasattr(sim_result, 'U_DG'):
            dg_status = [sim_result.U_DG[i][t] for t in island_periods]
            failed_hours = sum(1 for s in dg_status if s == 0)
            print(f"  DG Unit {i}: Failed {failed_hours}/{len(island_periods)} hours")
            if failed_hours > 0:
                dg_failures += 1
    
    if dg_failures > 0:
        print(f"\n✗ DG FAILURE DETECTED: {dg_failures} units experienced failures during island mode")
        print("  → This explains why DG never activated!")
    
    # Check demand scaling (tier mode)
    print(f"\n【DEMAND SCALING (TIER MODE)】")
    tier_3_hours = sum(1 for t in island_periods if 'tier_3' in sim_result.ems_mode[t].lower())
    tier_2_hours = sum(1 for t in island_periods if 'tier_2' in sim_result.ems_mode[t].lower())
    tier_1_hours = sum(1 for t in island_periods if 'tier_1' in sim_result.ems_mode[t].lower())
    
    print(f"  Tier 1 hours (100% demand): {tier_1_hours}")
    print(f"  Tier 2 hours (60% demand): {tier_2_hours}")
    print(f"  Tier 3 hours (30% demand): {tier_3_hours}")
    
    if tier_3_hours > tier_1_hours:
        print(f"\n⚠️  CRITICAL: Most island hours are in TIER 3 (30% demand)")
        print("  → This drastically reduces effective demand")
        print("  → With low demand + some renewables, deficit may never trigger DG!")
    
    # Calculate actual deficit after tier scaling
    print(f"\n【ACTUAL DEFICIT AFTER TIER SCALING】")
    total_renewable = sum(sim_result.P_wt[t] + sim_result.P_pv[t] for t in island_periods)
    total_demand_full = sum(df_weather['demand'].iloc[t] for t in island_periods)
    total_demand_scaled = sum(sim_result.demand[t] for t in island_periods)
    total_battery = sum(sum(sim_result.P_discharge[i][t] for i in range(len(sim_result.P_discharge))) for t in island_periods)
    
    print(f"  Full demand (100%): {total_demand_full:.0f} kWh")
    print(f"  Scaled demand (after tier): {total_demand_scaled:.0f} kWh")
    print(f"  Scaling ratio: {total_demand_scaled/total_demand_full*100:.1f}%")
    print(f"  Total renewable supply: {total_renewable:.0f} kWh")
    print(f"  Total battery discharge: {total_battery:.0f} kWh")
    print(f"  Total supply: {total_renewable + total_battery:.0f} kWh")
    print(f"  Remaining deficit: {max(0, total_demand_scaled - total_renewable - total_battery):.0f} kWh")
    
    if (total_renewable + total_battery) >= total_demand_scaled * 0.8:
        print(f"\n✓ CONCLUSION: Renewable + Battery covers ≥80% of scaled demand")
        print("  → DG is not needed even at low SOC!")
        print("  → This is the PRIMARY reason DG never activates")
else:
    print("\n✗ No island mode detected in simulation results")
