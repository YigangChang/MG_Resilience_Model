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
    EMSPolicy,
)
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.cost import evaluate_designs, compare_baseline_and_strategies
from microgrid.plotting import (
    plot_all_figures,
    plot_monte_carlo_results,
    create_design_side_by_side_comparisons,
)
from microgrid.monte_carlo import MonteCarloAnalyzer, MonteCarloConfig, UncertaintyDistribution

# ✨ 你新增的工具函式
from microgrid.weather_utils import build_time_input, build_hazard_from_weather


def build_default_monte_carlo_config(num_simulations: int = 300) -> MonteCarloConfig:
    """建立 Monte Carlo 預設配置（可調整模擬次數）"""
    return MonteCarloConfig(
        p_fail_WT=UncertaintyDistribution(
            name="WT_failure_probability",
            mean=0.05,
            std_dev=0.02,
            min_val=0.0,
            max_val=0.20,
            distribution_type="normal",
        ),
        p_fail_PV=UncertaintyDistribution(
            name="PV_failure_probability",
            mean=0.03,
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal",
        ),
        p_fail_DG=UncertaintyDistribution(
            name="DG_failure_probability",
            mean=0.02,
            std_dev=0.01,
            min_val=0.0,
            max_val=0.10,
            distribution_type="normal",
        ),
        p_fail_BAT=UncertaintyDistribution(
            name="BAT_failure_probability",
            mean=0.04,
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal",
        ),
        MTTR_WT=UncertaintyDistribution(
            name="WT_MTTR",
            mean=72.0,
            std_dev=36.0,
            min_val=24.0,
            max_val=240.0,
            distribution_type="lognormal",
        ),
        MTTR_PV=UncertaintyDistribution(
            name="PV_MTTR",
            mean=48.0,
            std_dev=24.0,
            min_val=12.0,
            max_val=144.0,
            distribution_type="lognormal",
        ),
        MTTR_DG=UncertaintyDistribution(
            name="DG_MTTR",
            mean=36.0,
            std_dev=18.0,
            min_val=12.0,
            max_val=120.0,
            distribution_type="lognormal",
        ),
        MTTR_BAT=UncertaintyDistribution(
            name="BAT_MTTR",
            mean=24.0,
            std_dev=12.0,
            min_val=6.0,
            max_val=72.0,
            distribution_type="lognormal",
        ),
        fuel_storage_uncertainty=UncertaintyDistribution(
            name="fuel_storage_capacity",
            mean=1.0,
            std_dev=0.15,
            min_val=0.70,
            max_val=1.30,
            distribution_type="normal",
        ),
        fuel_consumption_rate_uncertainty=UncertaintyDistribution(
            name="fuel_consumption_rate",
            mean=1.0,
            std_dev=0.20,
            min_val=0.60,
            max_val=1.50,
            distribution_type="normal",
        ),
        num_simulations=num_simulations,
        random_seed=42,
    )


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
            ems_policy=ems_policy,
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
        ems_policy=ems_policy,
    )

    rbcr_results = compare_baseline_and_strategies(
        results=results,
        cost=cost_params,
        scenario=scenario_hurricane,
        baseline_index=0,
    )

    # ============================================================
    # 7. Monte Carlo（每個 Design）
    # ============================================================
    print("\n===== Monte Carlo Comparison (per design) =====")
    os.makedirs("charts/monte_carlo", exist_ok=True)

    num_mc_simulations = 300
    mc_analyzer = MonteCarloAnalyzer(
        config=build_default_monte_carlo_config(num_simulations=num_mc_simulations)
    )

    mc_results = []
    for i, design in enumerate(designs):
        print(f"\nRunning MC for Design {i} ({num_mc_simulations} simulations)...")
        mc_result = mc_analyzer.run_analysis(
            design=design,
            scenario=scenario_hurricane,
            time_input=time_input,
            critical_load_ratio=0.2,
            ems_policy=ems_policy,
        )
        mc_results.append(mc_result)

        # 每個 design 保存獨立統計
        mc_summary_df = mc_result.to_dataframe()
        mc_summary_df.to_csv(
            f"charts/monte_carlo/MC_Design_{i}_Summary.csv",
            index=False,
        )

        # 每個 design 繪圖
        plot_monte_carlo_results(
            mc_result,
            output_dir=f"charts/monte_carlo/design_{i}",
        )

    # 合併輸出：確定性 + Monte Carlo 比較
    combined_rows = []
    for i, (det_res, mc_res) in enumerate(zip(results, mc_results)):
        combined_rows.append({
            "Design": i,
            "Deterministic_EENS": det_res["EENS"],
            "Deterministic_EID": det_res["EID"],
            "Deterministic_NPR": det_res["NPR"],
            "Deterministic_CLSR": det_res["CLSR"],
            "Deterministic_fuel_used_gal": det_res["fuel_used"],
            "MC_NPR_mean": mc_res.NPR_mean,
            "MC_NPR_std": mc_res.NPR_std,
            "MC_NPR_p05": mc_res.NPR_percentile_5,
            "MC_NPR_p95": mc_res.NPR_percentile_95,
            "MC_CLSR_mean": mc_res.CLSR_mean,
            "MC_CLSR_std": mc_res.CLSR_std,
            "MC_EID_mean": mc_res.EID_mean,
            "MC_EID_std": mc_res.EID_std,
            "MC_fuel_consumed_mean_gal": mc_res.fuel_consumed_mean,
            "MC_fuel_shortage_probability": mc_res.fuel_shortage_probability,
        })

    combined_df = pd.DataFrame(combined_rows)
    combined_path = "charts/monte_carlo/Designs_Deterministic_vs_MC_Comparison.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined comparison CSV: {combined_path}")

    # ============================================================
    # 8. 印結果
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

    print(f"\nAll Designs Monte Carlo Summary:")
    for i, mc_res in enumerate(mc_results):
        print(
            f"  Design {i}: NPR={mc_res.NPR_mean:.4f}±{mc_res.NPR_std:.4f}, "
            f"CLSR={mc_res.CLSR_mean:.4f}±{mc_res.CLSR_std:.4f}, "
            f"FuelShortageProb={mc_res.fuel_shortage_probability:.2%}"
        )
    
    print(f"\nBaseline fuel sustainability (h): {sim_result.fuel_sustainability_h}")
    # ============================================================
    # 9. 繪圖（每個 Design 各自輸出）
    # ============================================================
    for i, design_sim_result in enumerate(sim_results):
        design_output_dir = f"charts/design_{i}"
        print(f"\nPlotting deterministic figures for Design {i} -> {design_output_dir}")
        plot_all_figures(
            df=df_weather,
            sim_result=design_sim_result,
            scenario=scenario_hurricane,
            disturbance_start=disturbance_start,
            disturbance_end=disturbance_end,
            output_dir=design_output_dir,
        )

    create_design_side_by_side_comparisons(
        base_dir="charts",
        output_dir="charts/design_comparison",
    )

    # ============================================================
    # 10. 【調試】輸出 8 月 9-10 日電池放電資料
    # ============================================================
    if len(sim_results) > 1:
        print("\n" + "=" * 110)
        print("【調試：Design 1 在 8 月 9-10 日的電池放電逐時資料】")
        print("=" * 110)
        
        sim_result_d1 = sim_results[1]  # Design 1
        
        # 找出 8 月 9 日到 8 月 10 日的時間索引
        idx_list = []
        for t in range(len(df_weather)):
            ts_str = str(df_weather['datetime'].iloc[t])
            if '2009-08-09' in ts_str or '2009-08-10' in ts_str:
                idx_list.append(t)
        
        if idx_list:
            idx_start, idx_end = idx_list[0], idx_list[-1]
            print(f"\n時間範圍: {df_weather['datetime'].iloc[idx_start]} 到 {df_weather['datetime'].iloc[idx_end]}")
            print(f"時間索引: t={idx_start} 到 t={idx_end} (共 {idx_end - idx_start + 1} 小時)\n")
            
            print(f"{'時間':<20} {'WT':>8} {'PV':>8} {'DG':>8} {'放電':>8} {'需求':>8} {'供應':>8} {'缺電':>8} {'SOC':>6} {'模式':<20}")
            print("-" * 120)
            
            total_discharge = 0.0
            total_demand = 0.0
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
                
                print(f"{time_str:<20} {p_wt:>8.0f} {p_pv:>8.0f} {p_dg:>8.0f} {p_discharge:>8.1f} {demand:>8.1f} {served:>8.1f} {unserved:>8.1f} {avg_soc:>6.3f} {ems_mode:<20}")
            
            print("-" * 120)
            print(f"\n電池放電統計 (8月9-10日):")
            print(f"  總放電量: {total_discharge:.1f} kWh")
            print(f"  平均每小時: {total_discharge / (idx_end - idx_start + 1):.1f} kW")
            print(f"  總需求: {total_demand:.1f} kWh")
            print(f"  放電佔比: {total_discharge/total_demand*100:.1f}%")
