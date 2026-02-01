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
from microgrid.cost import evaluate_designs, compare_baseline_and_strategies
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
        # Failure probabilities
        base_p_damage_WT=0.10,
        base_p_damage_PV=0.05,
        base_p_damage_DG=0.01,
        base_p_damage_BAT=0.01,
        # Mean time to repair (hours)
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
    # 4. 微電網設計
    # ============================================================
    
    def compute_fuel_storage(
    fuel_rate_max: float,
    days_fuel: int,
    P_DG: list,
) -> float:
        """
        Compute required fuel storage based on number of DG units.
        fuel_rate_max: gal/hr per DG
        days_fuel: fuel autonomy in days
        P_DG: list of DG capacities
        """
        n_dg = len(P_DG)
        return fuel_rate_max * 24 * days_fuel * n_dg

    
    B_max_list = [4000.0] * 5
    B_init_list = [b * 0.8 for b in B_max_list]

    fuel_rate_max = 350.0  # gal/hr per DG
    days_fuel = 7

    designs = []

    # =========================
    # 0️⃣ Baseline design
    # =========================
    P_DG_0 = [5000.0] * 1

    designs.append(
        MicrogridDesign(
            P_WT=[2000.0] * 5,
            P_PV=[1000.0] * 20,
            P_DG=P_DG_0,
            P_BAT=[2000.0] * 1,
            B_max=B_max_list,
            B_init=B_init_list,
            eta_c=0.97,
            eta_d=0.98,
            C_WT=0.22,
            C_PV=0.36,
            A_WT=0.98,
            fuel_rate_max=fuel_rate_max,
            fuel_storage=compute_fuel_storage(
                fuel_rate_max, days_fuel, P_DG_0
            ),
            DG_min_loading=0.2,
            DG_max_loading=0.8,
            B_min_soc_frac=0.2,
            B_max_soc_frac=0.8,
            C_rate_charge=1.0,
            C_rate_discharge=1.0,
        )
    )

    # =========================
    # 1️⃣ Strategy: add more batteries
    # =========================
    P_DG_1 = [5000.0] * 1

    designs.append(
        MicrogridDesign(
            P_WT=[2000.0] * 5,
            P_PV=[1000.0] * 20,
            P_DG=P_DG_1,
            P_BAT=[2000.0] * 5,
            B_max=B_max_list,
            B_init=B_init_list,
            eta_c=0.97,
            eta_d=0.98,
            C_WT=0.22,
            C_PV=0.36,
            A_WT=0.98,
            fuel_rate_max=fuel_rate_max,
            fuel_storage=compute_fuel_storage(
                fuel_rate_max, days_fuel, P_DG_1
            ),
            DG_min_loading=0.2,
            DG_max_loading=0.8,
            B_min_soc_frac=0.2,
            B_max_soc_frac=0.8,
            C_rate_charge=1.0,
            C_rate_discharge=1.0,
        )
    )

    # =========================
    # 2️⃣ Strategy: add more batteries + more diesels
    # =========================
    P_DG_2 = [5000.0] * 2

    designs.append(
        MicrogridDesign(
            P_WT=[2000.0] * 5,
            P_PV=[1000.0] * 20,
            P_DG=P_DG_2,
            P_BAT=[2000.0] * 5,
            B_max=B_max_list,
            B_init=B_init_list,
            eta_c=0.97,
            eta_d=0.98,
            C_WT=0.22,
            C_PV=0.36,
            A_WT=0.98,
            fuel_rate_max=fuel_rate_max,
            fuel_storage=compute_fuel_storage(
                fuel_rate_max, days_fuel, P_DG_2
            ),
            DG_min_loading=0.2,
            DG_max_loading=0.8,
            B_min_soc_frac=0.2,
            B_max_soc_frac=0.8,
            C_rate_charge=1.0,
            C_rate_discharge=1.0,
        )
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
        VOLL=1000.0, #Value of Lost Load ($/kWh)
        C_fix_baseline=1000000.0, #Baseline fixed cost ($/year)
        C_fix_strategy=1200000.0,    # $/year (battery + DG annualized)
        fuel_price_per_gal=3.5, #Fuel price ($/gal)
        #planning_horizon_years=10,
        #wacc=0.05, 
    )

    # ============================================================
    # 5. 模擬
    # ============================================================
    # `designs` is a list — run simulation for each design and collect results
    sim_results = []
    for idx, design in enumerate(designs):
        sim = simulate_microgrid_resilience(
            design=design,
            scenario=scenario_hurricane,
            time_input=time_input,
            critical_load_ratio=0.2,
            random_seed=42 + idx,
        )
        sim_results.append(sim)

    # keep compatibility with existing downstream code: use first design as baseline
    sim_result = sim_results[0]

    # 列印每個 design 的模擬摘要（EENS / EENS_ratio / LOLE / fuel_used）
    for i, sim in enumerate(sim_results):
        print(f"--- Design {i} summary ---")
        print(
            f"EENS: {sim.EENS}, EENS_ratio: {sim.EENS_ratio}, LOLE: {sim.LOLE}, "
            f"critical_load_survival_time: {sim.critical_load_survival_time}, fuel_used: {sim.fuel_used}"
        )

    print("DG total output after tfr:",
          sum(sim_result.P_dg[disturbance_end+1 : disturbance_end+48]))

    # ============================================================
    # 6. Cost 評估
    # ============================================================

    results = evaluate_designs(
        designs=designs,
        scenario=scenario_hurricane,
        time_input=time_input,
        cost=cost_params,
        days_in_dataset=days_in_dataset,
    )

    rbcr_results = compare_baseline_and_strategies(
        results=results,
        cost=cost_params,
        scenario=scenario_hurricane,
        baseline_index=0,
    )

    # ============================================================
    # 7. 印結果
    # ============================================================
    print("===== Microgrid Resilience & Cost Result (per design) =====")
    
    # Print baseline (Design 0) resilience metrics
    print(f"\nBaseline Design (Design 0):")
    print(f"  EENS: {results[0]['EENS']:.1f}")
    print(f"  EID: {results[0]['EID']:.1f}")
    print(f"  NPR: {results[0]['NPR']:.3f}")
    print(f"  CPSO: {results[0]['CPSO']:.4f}")
    print(f"  CPH: {results[0]['CPH']:.4f}")
    print(f"  EARC: {results[0]['EARC']:.4f}")
    print(f"  Fuel used: {results[0]['fuel_used']:.1f} gal")
    
    # Print comparison results (strategies vs baseline)
    if len(rbcr_results) > 0:
        print(f"\nStrategy Comparison (vs Baseline):")
        for comp in rbcr_results:
            strategy_idx = comp['strategy_index']
            a_eens = comp['A_EENS'] if comp['A_EENS'] is not None else 0.0
            rbcr = comp['RBCR'] if comp['RBCR'] is not None else 0.0
            print(f"  Design {strategy_idx} vs Design {comp['baseline_index']}:")
            print(f"    A_EENS: {a_eens:.4f}")
            print(f"    RBCR: {rbcr:.4f}")
    
    # Print all design summaries
    print(f"\nAll Designs Summary:")
    for i, res in enumerate(results):
        print(f"  Design {i}: EENS={res['EENS']:.1f}, fuel_used={res['fuel_used']:.1f}, CPSO={res['CPSO']:.4f}, CPH={res['CPH']:.4f}, EARC={res['EARC']:.4f}")
    
    print(f"\nBaseline fuel sustainability (h): {sim_result.fuel_sustainability_h}")
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
