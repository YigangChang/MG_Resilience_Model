"""
調試腳本：輸出 Design 1 在 8 月 9-10 日期間的電池放電逐時資料
"""
import os
import pandas as pd
import numpy as np

from microgrid.models import (
    DisturbanceScenario,
    MicrogridDesign,
    CostParameters,
    EMSPolicy,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather


def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    """根據 DG 數量計算所需燃油儲存量"""
    n_dg = len(P_DG)
    return fuel_rate_max * 24 * days_fuel * n_dg


# ============================================================
# 讀取資料
# ============================================================
df_weather = pd.read_csv("Morakot_weather_with_demand.csv")
df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
df_weather["timestamp"] = df_weather["datetime"]

demand_series = df_weather["demand"].tolist()
N = len(df_weather)
days_in_dataset = N / 24.0

print(f"總資料長度: {N} 小時（{days_in_dataset:.1f} 天）")
print(f"時間範圍: {df_weather['datetime'].iloc[0]} 到 {df_weather['datetime'].iloc[-1]}")

# ============================================================
# 建立時間序列和危險值
# ============================================================
time_input = build_time_input(df_weather, demand_series)
hazard = build_hazard_from_weather(df_weather)

# ============================================================
# 定義災害情景
# ============================================================
td_ts = pd.Timestamp("2009-08-07 00:00:00")
tfr_ts = pd.Timestamp("2009-08-10 00:00:00")

try:
    disturbance_start = df_weather.index[df_weather["datetime"] == td_ts][0]
    disturbance_end = df_weather.index[df_weather["datetime"] == tfr_ts][0]
except IndexError:
    disturbance_start = (df_weather["datetime"] - td_ts).abs().idxmin()
    disturbance_end = (df_weather["datetime"] - tfr_ts).abs().idxmin()

print(f"\n災害時間:")
print(f"  災害開始: t={disturbance_start}, {df_weather['datetime'].iloc[disturbance_start]}")
print(f"  災害結束: t={disturbance_end}, {df_weather['datetime'].iloc[disturbance_end]}")

scenario_hurricane = DisturbanceScenario(
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
# 定義 Design 1（增強版）
# ============================================================
P_DG_1 = [500.0] * 1

design_1 = MicrogridDesign(
    P_WT=[400.0],
    P_PV=[300.0],
    P_DG=P_DG_1,
    P_BAT=[150.0],
    B_max=[500.0],
    B_init=[250.0],
    eta_c=0.95,
    eta_d=0.95,
    C_WT=0.35,
    C_PV=0.20,
    A_WT=1000.0,
    fuel_rate_max=50.0,
    fuel_storage=compute_fuel_storage(50.0, 7, P_DG_1),
    DG_min_loading=0.2,
    DG_max_loading=1.0,
    B_min_soc_frac=0.2,
    B_max_soc_frac=0.8,
    C_rate_charge=0.5,
    C_rate_discharge=0.5,
)

# ============================================================
# 定義 EMS 策略
# ============================================================
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
# 執行模擬
# ============================================================
print("\n執行 Design 1 模擬...")
sim_result = simulate_microgrid_resilience(
    design=design_1,
    scenario=scenario_hurricane,
    time_input=time_input,
    critical_load_ratio=0.2,
    random_seed=42,
    ems_policy=ems_policy,
)

# ============================================================
# 找出 8 月 9 日到 8 月 10 日的時間索引
# ============================================================
aug_9_start = pd.Timestamp("2009-08-09 00:00:00")
aug_10_end = pd.Timestamp("2009-08-10 23:59:59")

idx_start = (df_weather["datetime"] - aug_9_start).abs().idxmin()
idx_end = (df_weather["datetime"] - aug_10_end).abs().idxmin()

print(f"\n8 月 9-10 日時間範圍:")
print(f"  開始: t={idx_start}, {df_weather['datetime'].iloc[idx_start]}")
print(f"  結束: t={idx_end}, {df_weather['datetime'].iloc[idx_end]}")

# ============================================================
# 提取該時段的電池放電資料
# ============================================================
print(f"\n該時段電池放電逐時資料（按電池單元）:")
print("=" * 100)

output_rows = []
for t in range(idx_start, idx_end + 1):
    time_str = df_weather['datetime'].iloc[t].strftime("%Y-%m-%d %H:%M:%S")
    hour = df_weather['datetime'].iloc[t].hour
    demand = sim_result.demand[t]
    p_wt = sim_result.P_wt[t]
    p_pv = sim_result.P_pv[t]
    p_dg = sim_result.P_dg[t]
    served = sim_result.Gt[t]
    unserved = sim_result.Tt[t]
    
    # 電池放電（總和）
    p_discharge_total = sum(sim_result.P_discharge[i][t] for i in range(len(sim_result.P_discharge)))
    
    # 電池 SOC
    avg_soc = sim_result.avg_soc_frac[t]
    
    output_rows.append({
        'Time': time_str,
        'Hour': hour,
        'WT_kW': round(p_wt, 1),
        'PV_kW': round(p_pv, 1),
        'DG_kW': round(p_dg, 1),
        'BatDischarge_kW': round(p_discharge_total, 1),
        'Demand_kW': round(demand, 1),
        'Served_kW': round(served, 1),
        'Unserved_kW': round(unserved, 1),
        'Avg_SOC_frac': round(avg_soc, 3),
        'EMS_Mode': sim_result.ems_mode[t],
    })

output_df = pd.DataFrame(output_rows)

# 打印表格
print(output_df.to_string(index=False))

# 儲存到 CSV
output_csv_path = "debug_battery_discharge_aug_9_10.csv"
output_df.to_csv(output_csv_path, index=False)
print(f"\n已保存到: {output_csv_path}")

# ============================================================
# 統計該時段的電池放電總量
# ============================================================
total_discharge = output_df['BatDischarge_kW'].sum()
print(f"\n8 月 9-10 日電池放電總量: {total_discharge:.1f} kWh")
print(f"平均每小時放電量: {total_discharge / len(output_df):.1f} kW")
print(f"最大放電功率: {output_df['BatDischarge_kW'].max():.1f} kW")
print(f"最小放電功率: {output_df['BatDischarge_kW'].min():.1f} kW")

# ============================================================
# 檢查該時段的電池 SOC 變化
# ============================================================
print(f"\n電池 SOC 變化:")
print(f"  開始 (8月9日00時): {output_df['Avg_SOC_frac'].iloc[0]:.3f}")
print(f"  結束 (8月10日23時): {output_df['Avg_SOC_frac'].iloc[-1]:.3f}")
print(f"  最低: {output_df['Avg_SOC_frac'].min():.3f}")
print(f"  最高: {output_df['Avg_SOC_frac'].max():.3f}")

print("\n✓ 調試完成")
