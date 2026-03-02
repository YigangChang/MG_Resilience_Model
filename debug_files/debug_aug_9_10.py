#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified debug script: Extract Aug 9-10 battery discharge data for Design 1
without running Monte Carlo
"""
import os
import numpy as np
import pandas as pd
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(__file__))

from microgrid.models import (
    MicrogridDesign,
    CostParameters,
    EMSPolicy,
    TimeSeriesInput,
    DisturbanceScenario,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather

# ============================================================
# 1. Load weather data
# ============================================================
print("[1] Loading weather data...")
df_weather = pd.read_csv('Morakot_weather_with_demand.csv')
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
print(f"  Loaded {len(df_weather)} hours of data")
print(f"  Date range: {df_weather['datetime'].min()} to {df_weather['datetime'].max()}")

# ============================================================
# 2. Setup simulation parameters
# ============================================================
print("\n[2] Setting up designs...")

# Read Design parameters from run_simulation.py
# Design 0, 1, 2
P_WT_0 = [2000.0] * 5
P_PV_0 = [1000.0] * 20
P_DG_0 = [5000.0] * 1
P_BAT_0 = [2000.0] * 1

P_WT_1 = [1000.0] * 2
P_PV_1 = [2500.0] * 20
P_DG_1 = [5000.0] * 1
P_BAT_1 = [4000.0] * 2

P_WT_2 = [2500.0] * 10
P_PV_2 = [1000.0] * 20
P_DG_2 = [5000.0] * 1
P_BAT_2 = [4000.0] * 2

# Battery parameters
days_fuel = 20
fuel_rate_max = 50.0

def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    return fuel_rate_max * 24 * days_fuel * (sum(P_DG) / 5000.0) * 1.333  # Convert kW to L/h

B_max_list = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]  #maximum battery capacity (kWh)
B_init_list = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0] #initial battery state of charge (kWh)

designs = [
    MicrogridDesign(
        P_WT=P_WT_0,
        P_PV=P_PV_0,
        P_DG=P_DG_0,
        P_BAT=P_BAT_0,
        B_max=B_max_list,
        B_init=B_init_list,
        eta_c=0.97,
        eta_d=0.98,
        C_WT=0.22,
        C_PV=0.36,
        A_WT=0.98,
        fuel_rate_max=fuel_rate_max,
        fuel_storage=compute_fuel_storage(fuel_rate_max, days_fuel, P_DG_0),
        DG_min_loading=0.2,
        DG_max_loading=0.8,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=1.0,
        C_rate_discharge=1.0,
    ),
    MicrogridDesign(
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
        fuel_rate_max=fuel_rate_max,
        fuel_storage=compute_fuel_storage(fuel_rate_max, days_fuel, P_DG_1),
        DG_min_loading=0.2,
        DG_max_loading=0.8,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=1.0,
        C_rate_discharge=1.0,
    ),
    MicrogridDesign(
        P_WT=P_WT_2,
        P_PV=P_PV_2,
        P_DG=P_DG_2,
        P_BAT=P_BAT_2,
        B_max=B_max_list,
        B_init=B_init_list,
        eta_c=0.97,
        eta_d=0.98,
        C_WT=0.22,
        C_PV=0.36,
        A_WT=0.98,
        fuel_rate_max=fuel_rate_max,
        fuel_storage=compute_fuel_storage(fuel_rate_max, days_fuel, P_DG_2),
        DG_min_loading=0.2,
        DG_max_loading=0.8,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=1.0,
        C_rate_discharge=1.0,
    ),
]

# ============================================================
# 3. Setup cost and EMS parameters
# ============================================================
cost_params = CostParameters(
    I_WT=[81_100_000.0]*5,
    I_PV=[57_500_000.0]*20,
    I_DG=[109_000_000.0],
    I_BAT=[35_000_000.0],
    M_WT=[3_100_000.0]*5,
    M_PV=[860_000.0]*20,
    M_DG=[3_300_000.0],
    M_BAT=[700_000.0],
    H_WT=[24_000_000.0]*5,
    H_PV=[20_000_000.0]*20,
    H_DG=[16_000_000.0],
    H_BAT=[5_200_000.0],
    VOLL=1000.0,
    C_fix_baseline=1000000.0,
    C_fix_strategy=1200000.0,
    fuel_price_per_gal=3.5,
)

ems_policy = EMSPolicy(
    pre_event_hours=24,
    pre_event_target_soc=0.90,
    pre_event_soc_max=0.90,
    dg_start_soc=0.30,
    dg_stop_soc=0.70,
    load_tier_multipliers=[1.0, 0.6, 0.3],
    load_tier_soc_thresholds=[0.5, 0.3],
)

# ============================================================
# 4. Create time input and hazard
# ============================================================
print("\n[3] Building time input...")
demand_series = df_weather["demand"].tolist()
N = len(df_weather)
days_in_dataset = N / 24.0
time_input = build_time_input(df_weather, demand_series)

print("\n[4] Building hazard profile...")
hazard = build_hazard_from_weather(df_weather)

# Find disturbance indices
td_ts = pd.Timestamp("2009-08-07 00:00:00")
tfr_ts = pd.Timestamp("2009-08-10 00:00:00")

try:
    disturbance_start = df_weather.index[df_weather["datetime"] == td_ts][0]
    disturbance_end = df_weather.index[df_weather["datetime"] == tfr_ts][0]
except IndexError:
    disturbance_start = (df_weather["datetime"] - td_ts).abs().idxmin()
    disturbance_end = (df_weather["datetime"] - tfr_ts).abs().idxmin()

scenario = DisturbanceScenario(
    name="Morakot_2009",
    disturbance_start=disturbance_start,
    disturbance_end=disturbance_end,
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

# ============================================================
# 5. Run simulations for all designs
# ============================================================
print("\n[5] Running simulations...")
results = []
for i, design in enumerate(designs):
    print(f"  Running Design {i}...")
    result = simulate_microgrid_resilience(
        design=design,
        scenario=scenario,
        time_input=time_input,
        critical_load_ratio=0.2,
        random_seed=42 + i,
        ems_policy=ems_policy,
    )
    results.append(result)
    print(f"    EENS={result.EENS:.1f}, Fuel used={result.fuel_used:.1f} L")

# ============================================================
# 6. Extract Aug 9-10 data for Design 1
# ============================================================
print("\n" + "="*120)
print("【調試：Design 1 在 8 月 9-10 日的電池放電逐時資料】")
print("="*120)

sim_result_d1 = results[1]  # Design 1

# Find Aug 9-10 indices
idx_list = []
for t in range(len(df_weather)):
    ts_str = str(df_weather['datetime'].iloc[t])
    if '2009-08-09' in ts_str or '2009-08-10' in ts_str:
        idx_list.append(t)

if idx_list:
    idx_start, idx_end = idx_list[0], idx_list[-1]
    print(f"\n時間範圍: {df_weather['datetime'].iloc[idx_start]} 到 {df_weather['datetime'].iloc[idx_end]}")
    print(f"時間索引: t={idx_start} 到 t={idx_end} (共 {idx_end - idx_start + 1} 小時)\n")
    
    print(f"{'時間':<20} {'WT(kW)':>8} {'PV(kW)':>8} {'DG(kW)':>8} {'放電(kW)':>10} {'需求(kW)':>10} {'供應(kW)':>10} {'缺電(kW)':>10} {'SOC':>6} {'模式':<20}")
    print("-"*140)
    
    total_discharge = 0.0
    total_demand = 0.0
    total_served = 0.0
    total_dg = 0.0
    
    output_data = []
    
    for t in range(idx_start, idx_end + 1):
        time_str = str(df_weather['datetime'].iloc[t])[:16]
        p_wt = sim_result_d1.P_wt[t]
        p_pv = sim_result_d1.P_pv[t]
        p_dg = sim_result_d1.P_dg[t]
        p_discharge = sum(sim_result_d1.P_discharge[i][t] for i in range(len(sim_result_d1.P_discharge)))
        demand = sim_result_d1.demand[t]
        served = sim_result_d1.Gt[t]
        unserved = sim_result_d1.Tt[t]
        avg_soc = sim_result_d1.avg_soc_frac[t]
        ems_mode = sim_result_d1.ems_mode[t]
        
        total_discharge += p_discharge
        total_demand += demand
        total_served += served
        total_dg += p_dg
        
        output_data.append({
            'datetime': time_str,
            'WT': p_wt,
            'PV': p_pv,
            'DG': p_dg,
            'Discharge': p_discharge,
            'Demand': demand,
            'Served': served,
            'Unserved': unserved,
            'SOC': avg_soc,
            'EMS_Mode': ems_mode,
        })
        
        print(f"{time_str:<20} {p_wt:>8.1f} {p_pv:>8.1f} {p_dg:>8.1f} {p_discharge:>10.1f} {demand:>10.1f} {served:>10.1f} {unserved:>10.1f} {avg_soc:>6.3f} {ems_mode:<20}")
    
    print("-"*140)
    print(f"\n電池放電統計 (8月9-10日):")
    print(f"  總放電量: {total_discharge:.1f} kWh")
    print(f"  平均每小時: {total_discharge / (idx_end - idx_start + 1):.1f} kW")
    print(f"  總需求: {total_demand:.1f} kWh")
    print(f"  總供應: {total_served:.1f} kWh")
    print(f"  總缺電: {sum(output_data[i]['Unserved'] for i in range(len(output_data))):.1f} kWh")
    print(f"  放電佔總供應比: {total_discharge/total_served*100:.1f}%")
    print(f"  DG貢獻: {total_dg:.1f} kWh ({total_dg/total_served*100:.1f}%)")
    print(f"  WT貢獻: {sum(output_data[i]['WT'] for i in range(len(output_data))):.1f} kWh")
    print(f"  PV貢獻: {sum(output_data[i]['PV'] for i in range(len(output_data))):.1f} kWh")
    
    # Save to CSV
    df_output = pd.DataFrame(output_data)
    df_output.to_csv('Aug_9_10_Design1_Discharge_Debug.csv', index=False)
    print(f"\n✓ 數據已保存到 Aug_9_10_Design1_Discharge_Debug.csv")
else:
    print("未找到 8 月 9-10 日的數據")
