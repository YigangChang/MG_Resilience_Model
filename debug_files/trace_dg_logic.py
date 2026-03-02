#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETAILED DEBUG: Trace DG activation logic step-by-step
Since DG is available (green in graph) but never activates, 
we need to check the ACTUAL deficit after battery discharge attempts.
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
from microgrid.weather_utils import build_time_input, build_hazard_from_weather

# We need to modify the simulation to output DG activation diagnostics
# Let's manually trace through the logic for a few critical hours

print("=" * 150)
print("【DETAILED DG ACTIVATION LOGIC TRACE】")
print("=" * 150)

# Load weather and setup
df_weather = pd.read_csv('Morakot_weather_with_demand.csv')
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

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

# Now manually trace the logic for a critical hour
# Let's pick hour 940 (2009-08-07 04:00) where:
# - SOC should be low
# - DG should be available (green in graph)
# - There should be deficit

print("\n【MANUAL LOGIC TRACE FOR HOUR 940 (2009-08-07 04:00)】\n")

t = 940
hour_time = df_weather['datetime'].iloc[t]
print(f"Time: {hour_time}")
print(f"Hour index: {t}")

# Check demand
full_demand = df_weather['demand'].iloc[t]
print(f"\n1. Full demand: {full_demand:.0f} kW")

# Simulate tier scaling (assuming tier_3 with 30% demand)
tier_multiplier = 0.30
effective_demand = full_demand * tier_multiplier
print(f"   Effective demand (tier 3, 30%): {effective_demand:.0f} kW")

# Check renewable generation
cf_wt = df_weather.get('cf_WT', pd.Series([0]*len(df_weather))).iloc[t]
cf_pv = df_weather.get('cf_PV', pd.Series([0]*len(df_weather))).iloc[t]
print(f"\n2. Capacity factors:")
print(f"   WT CF: {cf_wt:.3f}")
print(f"   PV CF: {cf_pv:.3f}")

# Assume some WT/PV might be damaged, let's say 50% availability
total_wt_capacity = sum(P_WT_1)
total_pv_capacity = sum(P_PV_1)
wt_gen = total_wt_capacity * cf_wt * 0.5  # 50% availability
pv_gen = total_pv_capacity * cf_pv * 0.5
print(f"\n3. Renewable generation (assuming 50% availability):")
print(f"   WT generation: {wt_gen:.0f} kW")
print(f"   PV generation: {pv_gen:.0f} kW")
print(f"   Total renewable: {wt_gen + pv_gen:.0f} kW")

# Calculate initial deficit
initial_deficit = effective_demand - (wt_gen + pv_gen)
print(f"\n4. Initial deficit (before battery): {initial_deficit:.0f} kW")

# Check battery status (assuming SOC at minimum 20%)
soc_current = 0.20  # 20% SOC (at minimum)
soc_min_frac = 0.20
total_battery_capacity = sum(B_max_list)
total_battery_power = sum(P_BAT_1)

print(f"\n5. Battery status:")
print(f"   Current SOC: {soc_current*100:.0f}%")
print(f"   Minimum SOC: {soc_min_frac*100:.0f}%")
print(f"   At minimum SOC? {soc_current <= soc_min_frac}")

if soc_current <= soc_min_frac:
    battery_discharge = 0.0
    print(f"   → Battery CANNOT discharge (at minimum SOC)")
else:
    battery_discharge = min(initial_deficit, total_battery_power)
    print(f"   → Battery can discharge up to {battery_discharge:.0f} kW")

print(f"   Battery discharge: {battery_discharge:.0f} kW")

# Calculate deficit after battery
deficit_after_battery = initial_deficit - battery_discharge
print(f"\n6. Deficit after battery discharge: {deficit_after_battery:.0f} kW")

# Check DG activation conditions
print(f"\n7. DG activation conditions:")
print(f"   ✓ Grid available? NO (island mode)")
print(f"   ✓ SOC ({soc_current*100:.0f}%) ≤ start threshold ({ems_policy.dg_start_soc*100:.0f}%)? {soc_current <= ems_policy.dg_start_soc}")
print(f"   → dg_enabled = {soc_current <= ems_policy.dg_start_soc}")

dg_enabled = (soc_current <= ems_policy.dg_start_soc)

print(f"\n   ✓ Deficit after battery > 0? {deficit_after_battery > 0} ({deficit_after_battery:.0f} kW)")
print(f"   ✓ Fuel remaining > 0? YES (24,000 gal)")
print(f"   ✓ dg_enabled? {dg_enabled}")

# Check DG constraints
dg_capacity = sum(P_DG_1)
dg_min_output = dg_capacity * design_1.DG_min_loading
dg_max_output = dg_capacity * design_1.DG_max_loading

print(f"\n8. DG constraints:")
print(f"   DG capacity: {dg_capacity:.0f} kW")
print(f"   DG min output (20%): {dg_min_output:.0f} kW")
print(f"   DG max output (80%): {dg_max_output:.0f} kW")
print(f"   Deficit: {deficit_after_battery:.0f} kW")

if deficit_after_battery > 0 and dg_enabled:
    if deficit_after_battery < dg_min_output:
        print(f"   ⚠️  Deficit ({deficit_after_battery:.0f} kW) < DG minimum ({dg_min_output:.0f} kW)")
        print(f"   → DG would run at minimum: {dg_min_output:.0f} kW")
        dg_output = dg_min_output
    else:
        dg_output = min(deficit_after_battery, dg_max_output)
        print(f"   ✓ DG can supply: {dg_output:.0f} kW")
    
    print(f"\n【DG SHOULD ACTIVATE】 Output: {dg_output:.0f} kW")
else:
    print(f"\n【DG WILL NOT ACTIVATE】")
    if deficit_after_battery <= 0:
        print(f"   Reason: No deficit after battery discharge")
    elif not dg_enabled:
        print(f"   Reason: dg_enabled = False (SOC not low enough)")

# Now check if the issue is in the ACTUAL deficit calculation
print("\n" + "=" * 150)
print("【HYPOTHESIS: The issue might be in how 'deficit' is tracked in the code】")
print("=" * 150)

print("""
In simulation.py, the logic flow is:

1. Calculate renewable generation
2. Calculate initial surplus/deficit
3. If surplus > 0: charge battery
4. If deficit > 0: discharge battery
5. After battery discharge, if deficit > 0 AND dg_enabled: use DG

The KEY question: Is the 'deficit' variable being updated correctly after battery discharge?

Let me check the exact code structure...
""")

# Read the actual simulation code
with open('microgrid/simulation.py', 'r', encoding='utf-8') as f:
    sim_code = f.read()

# Find the DG activation section
import re
dg_section = re.search(r'# --- DG 補缺.*?(?=\n\s*# ---|\n\s*P_gen_total)', sim_code, re.DOTALL)
if dg_section:
    print("\n【ACTUAL DG ACTIVATION CODE】")
    print("```python")
    print(dg_section.group(0))
    print("```")

print("\n" + "=" * 150)
print("【KEY FINDING】")
print("=" * 150)
print("""
Looking at the code structure, I notice that the DG activation happens in island mode,
but the 'deficit' variable needs to be properly calculated AFTER battery discharge.

The variable name is likely 'deficit' or 'residual' after the battery discharge loop.

If this variable is NOT tracking the remaining deficit correctly, or if it's reset
somewhere, DG won't activate even when it should.

Let me trace the EXACT variable names in the island mode section...
""")

# Find island mode section
island_section = re.search(r'elif not grid_available:.*?# Update battery SOC', sim_code, re.DOTALL)
if island_section:
    print("\n【ISLAND MODE SECTION (condensed)】")
    lines = island_section.group(0).split('\n')
    for i, line in enumerate(lines[:100]):  # Show first 100 lines
        if any(keyword in line for keyword in ['deficit', 'DG', 'dg_enabled', 'P_dg', 'residual']):
            print(f"{i:3d}: {line}")
