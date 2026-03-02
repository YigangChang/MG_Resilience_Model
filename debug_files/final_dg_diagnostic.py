#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL DIAGNOSTIC: Print ALL variables at ONE critical hour to see exact state
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from microgrid.models import MicrogridDesign, EMSPolicy, DisturbanceScenario
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather

# Setup (same as before)
df_weather = pd.read_csv('Morakot_weather_with_demand.csv')
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

P_WT_1 = [1000.0] * 2
P_PV_1 = [2500.0] * 20
P_DG_1 = [5000.0] * 1
P_BAT_1 = [4000.0] * 2
B_max_list = [200.0] * 8
B_init_list = [100.0] * 8

design_1 = MicrogridDesign(
    P_WT=P_WT_1, P_PV=P_PV_1, P_DG=P_DG_1, P_BAT=P_BAT_1,
    B_max=B_max_list, B_init=B_init_list,
    eta_c=0.97, eta_d=0.98, C_WT=0.22, C_PV=0.36, A_WT=0.98,
    fuel_rate_max=50.0, fuel_storage=24000.0,
    DG_min_loading=0.2, DG_max_loading=0.8,
    B_min_soc_frac=0.2, B_max_soc_frac=0.8,
    C_rate_charge=1.0, C_rate_discharge=1.0,
)

demand_series = df_weather["demand"].tolist()
time_input = build_time_input(df_weather, demand_series)
hazard = build_hazard_from_weather(df_weather)

disturbance_start_ts = pd.to_datetime("2009-08-07T00:00:00")
disturbance_end_ts = pd.to_datetime("2009-08-10T00:00:00")
disturbance_start_idx = (df_weather["datetime"] - disturbance_start_ts).abs().idxmin()
disturbance_end_idx = (df_weather["datetime"] - disturbance_end_ts).abs().idxmin()

scenario = DisturbanceScenario(
    name="Morakot_2009", disturbance_start=disturbance_start_idx,
    disturbance_end=disturbance_end_idx,
    base_p_damage_WT=0.10, base_p_damage_PV=0.05,
    base_p_damage_DG=0.01, base_p_damage_BAT=0.01,
    MTTR_WT=72.0, MTTR_PV=31.0, MTTR_DG=289.0, MTTR_BAT=3.0,
    grid_MTTR_hours=48, hazard=hazard,
    evaluation_horizon_hours=168, annual_occurrence_rate=0.1,
)

ems_policy = EMSPolicy(dg_start_soc=0.30, dg_stop_soc=0.70)

# Run simulation
sim_result = simulate_microgrid_resilience(
    design=design_1, scenario=scenario,
    time_input=time_input, ems_policy=ems_policy,
)

print("="*120)
print("FINAL DIAGNOSTIC: Complete variable dump for critical hours")
print("="*120)

# Check several hours in detail
for t in [937, 940, 945, 950]:  # Island mode hours
    print(f"\n{'='*120}")
    print(f"HOUR t={t}: {df_weather['datetime'].iloc[t]}")
    print(f"{'='*120}")
    
    print(f"EMS Mode: {sim_result.ems_mode[t]}")
    
    print(f"\n[1. DEMAND]")
    print(f"  Full demand: {df_weather['demand'].iloc[t]:.0f} kW")
    print(f"  Effective demand (Dt): {sim_result.demand[t]:.0f} kW")
    
    print(f"\n[2. RENEWABLE GENERATION]")
    print(f"  WT generation: {sim_result.P_wt[t]:.0f} kW")
    print(f"  PV generation: {sim_result.P_pv[t]:.0f} kW")
    print(f"  Total renewable: {sim_result.P_wt[t] + sim_result.P_pv[t]:.0f} kW")
    
    print(f"\n[3. BATTERY STATUS]")
    print(f"  Average SOC: {sim_result.avg_soc_frac[t]*100:.1f}%")
    total_discharge = sum(sim_result.P_discharge[i][t] for i in range(len(sim_result.P_discharge)))
    print(f"  Battery discharge: {total_discharge:.0f} kW")
    
    print(f"\n[4. DIESEL GENERATOR]")
    print(f"  DG output: {sim_result.P_dg[t]:.0f} kW")
    
    # Calculate deficit manually
    deficit_before_battery = sim_result.demand[t] - (sim_result.P_wt[t] + sim_result.P_pv[t])
    deficit_after_battery = deficit_before_battery - total_discharge
    
    print(f"\n[5. POWER BALANCE]")
    print(f"  Deficit before battery: {deficit_before_battery:.0f} kW")
    print(f"  Battery contribution: {total_discharge:.0f} kW")
    print(f"  Deficit after battery: {deficit_after_battery:.0f} kW")
    print(f"  DG output: {sim_result.P_dg[t]:.0f} kW")
    print(f"  Served load: {sim_result.Gt[t]:.0f} kW")
    print(f"  Unserved load: {sim_result.Tt[t]:.0f} kW")
    
    print(f"\n[6. DG ACTIVATION CONDITIONS]")
    print(f"  Island mode: {('island' in sim_result.ems_mode[t].lower())}")
    print(f"  SOC ({sim_result.avg_soc_frac[t]*100:.0f}%) <= start_soc (30%): {sim_result.avg_soc_frac[t] <= 0.30}")
    print(f"  -> dg_enabled should be: {sim_result.avg_soc_frac[t] <= 0.30}")
    print(f"  Deficit after battery ({deficit_after_battery:.0f} kW) > 0: {deficit_after_battery > 0}")
    print(f"  Deficit exceeds DG min loading (1000 kW): {deficit_after_battery >= 1000}")
    
    if deficit_after_battery > 0 and sim_result.avg_soc_frac[t] <= 0.30:
        print(f"\n  WARNING: ALL CONDITIONS MET -> DG SHOULD ACTIVATE AT {min(4000, deficit_after_battery):.0f} kW")
        if sim_result.P_dg[t] == 0:
            print(f"  ERROR: BUT DG OUTPUT = 0 kW!")
            print(f"\n  POSSIBLE CAUSES:")
            print(f"    1. DG unit failed (U_DG[0] = 0)")
            print(f"    2. Fuel exhausted")
            print(f"    3. Logic error in code")
            print(f"    4. 'dg_enabled' flag not properly set")

print(f"\n\n{'='*120}")
print("SUMMARY OF FINDINGS")
print(f"{'='*120}")
total_dg = sum(sim_result.P_dg)
print(f"Total DG generation across all hours: {total_dg:.1f} kWh")

if total_dg == 0:
    print(f"\nERROR: DG NEVER ACTIVATED during entire simulation!")
    print(f"\nGiven that:")
    print(f"  - DG is available (shown in graph as green during hours ~936-1000)")
    print(f"  - SOC drops to 10% (below 30% threshold)")
    print(f"  - Deficit exists after battery discharge")
    print(f"  - Fuel is available (24,000 gal)")
    print(f"\nThe most likely causes are:")
    print(f"\n[HYPOTHESIS 1]: 'dg_enabled' flag is not being set correctly")
    print(f"  - The hysteresis logic might have a bug")
    print(f"  - Or 'dg_enabled' is being reset somewhere unexpectedly")
    print(f"\n[HYPOTHESIS 2]: The deficit calculation is incorrect")
    print(f"  - Maybe the variable 'deficit' is not tracking correctly")
    print(f"  - Or battery discharge is consuming all deficit (but at min SOC this shouldn't happen)")
    print(f"\n[HYPOTHESIS 3]: U_DG[0] is actually 0 even though graph shows green")
    print(f"  - Maybe the graph is showing a different simulation run")
    print(f"  - Or there's a mismatch between diagnostic hour indices and graph hour indices")
else:
    print(f"\nSUCCESS: DG activated successfully")
