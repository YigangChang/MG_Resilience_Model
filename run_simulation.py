# run_simulation.py
"""
Microgrid resilience simulation with:
- Normal mode / Pre-storm / Disaster mode
- Hazard-based failures
- IEEE-style resiliency curve
- LCOE / LCOED tied to simulated diesel use
"""

import os
import numpy as np
import pandas as pd

from microgrid.models import (
    HazardProfile,
    TimeSeriesInput,
    DisturbanceScenario,
    MicrogridDesign,
    CostParameters,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.cost import compute_LCOE, compute_LCOED
from microgrid.plotting import plot_all_figures

# ✨ 你新增的工具函式
from microgrid.weather_utils import build_time_input, build_hazard_from_weather


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)

    # ============================================================
    # 1. 讀取莫拉克颱風整合後的天氣 + 負載資料
    # ============================================================
    df_weather = pd.read_csv("Morakot_weather_with_demand.csv")
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    df_weather["timestamp"] = df_weather["datetime"]


    # demand_series 應該與 weather 長度一致
    demand_series = df_weather["demand"].tolist()
    N = len(df_weather)
    days_in_dataset = N / 24.0

    # ============================================================
    # 2. time_input（CF_WT, CF_PV, demand, hours）
    # ============================================================
    time_input = build_time_input(df_weather, demand_series)

    # ============================================================
    # 3. hazard（給 DisturbanceScenario）
    # ============================================================
    hazard = build_hazard_from_weather(df_weather)

    # 颱風開始與結束
    td_ts = pd.Timestamp("2009-08-07 00:00:00")
    tfr_ts = pd.Timestamp("2009-08-10 00:00:00")

    try:
        disturbance_start = df_weather.index[df_weather["datetime"] == td_ts][0]
        disturbance_end = df_weather.index[df_weather["datetime"] == tfr_ts][0]
    except IndexError:
        disturbance_start = (df_weather["datetime"] - td_ts).abs().idxmin()
        disturbance_end = (df_weather["datetime"] - tfr_ts).abs().idxmin()

    # DisturbanceScenario
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
    )

    # ============================================================
    # 4. 微電網設計
    # ============================================================
    P_DG_units = [5000.0] * 1   # 1 台 5 MW  #!

    B_max_list = [4000.0] * 5     #!
    B_init_list = [b * 0.8 for b in B_max_list]

    fuel_rate_max = 350.0  # gal/hr
    days_fuel = 7     #!
    fuel_storage_required = fuel_rate_max * 24 * days_fuel * len(P_DG_units)

    design_example = MicrogridDesign(
        P_WT=[2000.0] * 5,
        P_PV=[1000.0] * 20,
        P_DG=P_DG_units,
        P_BAT=[2000.0] * 5,  #!
        B_max=B_max_list,
        B_init=B_init_list,
        eta_c=0.97,
        eta_d=0.98,
        C_WT=0.22,
        C_PV=0.36,
        A_WT=0.98,
        fuel_rate_max=fuel_rate_max,
        fuel_storage=fuel_storage_required,
        DG_min_loading=0.2,
        DG_max_loading=0.8,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=1.0,
        C_rate_discharge=1.0,
    )

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
        planning_horizon_years=10,
        wacc=0.05,
    )

    # ============================================================
    # 5. 模擬
    # ============================================================
    sim_result = simulate_microgrid_resilience(
        design=design_example,
        scenario=scenario_hurricane,
        time_input=time_input,
        critical_load_ratio=0.2,
        random_seed=42,
    )

    print("DG total output after tfr:",
          sum(sim_result.P_dg[disturbance_end+1 : disturbance_end+48]))

    # ============================================================
    # 6. LCOE / LCOED
    # ============================================================
    Ey_total = sum(sim_result.Gt)
    Dy_total = sum(sim_result.demand)

    scale_to_year = 365.0 / days_in_dataset
    Ey_year = [Ey_total * scale_to_year]
    Dy_year = [Dy_total * scale_to_year]

    LCOE_val = compute_LCOE(
        cost=cost_params,
        Ey_year=Ey_year,
        fuel_used_sim=sim_result.fuel_used,
        days_simulated=days_in_dataset,
        fuel_price_per_gal=97.0,
    )

    LCOED_val = compute_LCOED(
        cost=cost_params,
        Dy_year=Dy_year,
        fuel_used_sim=sim_result.fuel_used,
        days_simulated=days_in_dataset,
        fuel_price_per_gal=97.0,
    )

    # ============================================================
    # 7. 印結果
    # ============================================================
    print("===== Microgrid Resilience & Cost Result =====")
    print(f"Invulnerability         : {sim_result.invulnerability:.3f}")
    print(f"Resilience (curve)      : {sim_result.resilience_curve:.3f}")
    print(f"Recovery time [h]       : {sim_result.recovery_time_h}")
    print(f"Total fuel used [gal]   : {sim_result.fuel_used:.1f}")
    print(f"LCOE  [$ / kWh]         : {LCOE_val:.4f}")
    print(f"LCOED[$ / kWh]         : {LCOED_val:.4f}")
    print("EENS:", sim_result.EENS)
    print("EENS_ratio:", sim_result.EENS_ratio)
    print("LOLE:", sim_result.LOLE)
    print("Critical load survival (h):", sim_result.critical_load_survival_time)
    print("Fuel sustainability (h):", sim_result.fuel_sustainability_h)
    # ============================================================
    # 8. 繪圖
    # ============================================================
    plot_all_figures(
        df=df_weather,
        sim_result=sim_result,
        scenario=scenario_hurricane,
        disturbance_start=disturbance_start,
        disturbance_end=disturbance_end,
        output_dir="charts",
    )
