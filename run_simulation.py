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


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)

    # 讀取 5~11 月時間序列
    df = pd.read_csv("Microgrid_5to11_months_timeseries.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    demand_series = df["demand_kW"].tolist()
    cf_pv_series = df["pv_cf"].tolist()
    cf_wt_series = df["wt_cf"].tolist()
    hours_of_day = df["hour_of_day"].tolist()
    months_series = df["timestamp"].dt.month.tolist()
    days_series = df["timestamp"].dt.day.tolist()

    total_hours = len(df)
    days_in_dataset = total_hours / 24.0

    hazard = HazardProfile(
        time_h=list(range(total_hours)),
        wind_speed=df["wind_speed_m_s"].tolist(),
        rainfall=df["rain_mm_hr"].tolist(),
        solar_irradiance=df["solar_irr_ratio"].tolist(),
    )

    time_input = TimeSeriesInput(
        demand=demand_series,
        cf_WT=cf_wt_series,
        cf_PV=cf_pv_series,
        hours=hours_of_day,
    )

    # 找出 8/7 00:00 作為颱風開始時間
    disturbance_start = None
    for idx in range(total_hours):
        if months_series[idx] == 8 and days_series[idx] == 7 and hours_of_day[idx] == 0:
            disturbance_start = idx
            break
    if disturbance_start is None:
        raise RuntimeError("找不到 8/7 00:00 的 disturbance_start，請檢查時間序列資料。")

    disturbance_duration_h = 72  # 3 天
    disturbance_end = min(total_hours - 1, disturbance_start + disturbance_duration_h)

    scenario_hurricane = DisturbanceScenario(
        name="Morakot_like",
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
        hazard=hazard,
        evaluation_horizon_hours=168,
    )

    # 微電網設計：
    P_DG_units = [5000.0] * 1   # 1 台 5 MW

    B_max_list = [4000.0] * 1
    B_init_list = [b * 0.8 for b in B_max_list]  # 平時 80% SOC

    fuel_rate_max = 350.0  # [gal/hr] @ rated
    days_fuel = 3
    fuel_storage_required = fuel_rate_max * 24 * days_fuel * len(P_DG_units)

    design_example = MicrogridDesign(
        P_WT=[2000.0] * 5,
        P_PV=[1000.0] * 20,
        P_DG=P_DG_units,
        P_BAT=[2000.0] * 1,
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
        I_WT=[81_100_000.0],
        I_PV=[57_500_000.0],
        I_DG=[109_000_000.0],
        I_BAT=[35_000_000.0],
        M_WT=[3_100_000.0],
        M_PV=[860_000.0],
        M_DG=[3_300_000.0],
        M_BAT=[700_000.0],
        H_WT=[24_000_000.0],
        H_PV=[20_000_000.0],
        H_DG=[16_000_000.0],
        H_BAT=[5_200_000.0],
        planning_horizon_years=10,
        wacc=0.05,
    )

    # 跑模擬
    sim_result = simulate_microgrid_resilience(
        design=design_example,
        scenario=scenario_hurricane,
        time_input=time_input,
        critical_load_ratio=0.2,
        random_seed=42,
    )

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

    print("===== Microgrid Resilience & Cost Result =====")
    print(f"Invulnerability         : {sim_result.invulnerability:.3f}")
    print(f"Resilience (curve)      : {sim_result.resilience_curve:.3f}")
    print(f"Recovery time [h]       : {sim_result.recovery_time_h}")
    print(f"Total fuel used [gal]   : {sim_result.fuel_used:.1f}")
    print(f"LCOE  [$ / kWh]         : {LCOE_val:.4f}")
    print(f"LCOED[$ / kWh]         : {LCOED_val:.4f}")

    # 繪圖
    plot_all_figures(
        df=df,
        sim_result=sim_result,
        scenario=scenario_hurricane,
        disturbance_start=disturbance_start,
        disturbance_end=disturbance_end,
        output_dir="charts",
    )
