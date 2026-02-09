# run_simulation_with_monte_carlo.py
"""
微電網韌性模擬 + Monte Carlo 不確定分析整合
包含：
- 正常/災前/災害模式
- 基於危險值的設備故障
- IEEE 韌性曲線
- LCOE/LCOED 與模擬柴油使用相關
- Monte Carlo 不確定分析（新增）
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
    plot_ems_timeseries,
)
from microgrid.weather_utils import build_time_input, build_hazard_from_weather
from microgrid.monte_carlo import MonteCarloAnalyzer, MonteCarloConfig, UncertaintyDistribution


def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    """根據 DG 數量計算所需燃油儲存量"""
    n_dg = len(P_DG)
    return fuel_rate_max * 24 * days_fuel * n_dg


def run_deterministic_simulation(df_weather, time_input, hazard, scenario, designs, cost_params, ems_policy):
    """執行確定性微電網模擬（基準情景）"""
    print("=" * 80)
    print("【確定性模擬】")
    print("=" * 80)
    
    results = evaluate_designs(
        designs=designs,
        scenario=scenario,
        time_input=time_input,
        cost=cost_params,
        days_in_dataset=len(df_weather) / 24.0,
        ems_policy=ems_policy,
    )
    
    rbcr_results = compare_baseline_and_strategies(
        results=results,
        cost=cost_params,
        scenario=scenario,
        baseline_index=0,
    )
    
    print("\n【基準設計 (Design 0) 的韌性指標】")
    print(f"  EENS: {results[0]['EENS']:.1f} kWh")
    print(f"  EID: {results[0]['EID']:.1f} 小時")
    print(f"  NPR: {results[0]['NPR']:.3f}")
    print(f"  CLSR: {results[0]['CLSR']:.3f}")
    print(f"  CPSO: {results[0]['CPSO']:.4f}")
    print(f"  CPH: {results[0]['CPH']:.4f}")
    print(f"  EARC: {results[0]['EARC']:.4f}")
    print(f"  燃油消耗: {results[0]['fuel_used']:.1f} 加侖")
    
    if len(rbcr_results) > 0:
        print("\n【策略比較（相對於基準）】")
        for comp in rbcr_results:
            strategy_idx = comp['strategy_index']
            a_eens = comp['A_EENS'] if comp['A_EENS'] is not None else 0.0
            rbcr = comp['RBCR'] if comp['RBCR'] is not None else 0.0
            print(f"  Design {strategy_idx} vs Design {comp['baseline_index']}:")
            print(f"    A_EENS: {a_eens:.4f}")
            print(f"    RBCR: {rbcr:.4f}")
    
    return results, rbcr_results


def run_monte_carlo_analysis(design, scenario, time_input, ems_policy, num_simulations=1000):
    """執行 Monte Carlo 不確定分析"""
    print("\n" + "=" * 80)
    print("【Monte Carlo 不確定分析】")
    print("=" * 80)
    
    # 建立自訂配置
    mc_config = MonteCarloConfig(
        # 故障率分佈
        p_fail_WT=UncertaintyDistribution(
            name="WT_failure_probability",
            mean=0.05,
            std_dev=0.02,
            min_val=0.0,
            max_val=0.20,
            distribution_type="normal"
        ),
        p_fail_PV=UncertaintyDistribution(
            name="PV_failure_probability",
            mean=0.03,
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal"
        ),
        p_fail_DG=UncertaintyDistribution(
            name="DG_failure_probability",
            mean=0.02,
            std_dev=0.01,
            min_val=0.0,
            max_val=0.10,
            distribution_type="normal"
        ),
        p_fail_BAT=UncertaintyDistribution(
            name="BAT_failure_probability",
            mean=0.04,
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal"
        ),
        
        # 修復時間分佈 (對數常態)
        MTTR_WT=UncertaintyDistribution(
            name="WT_MTTR",
            mean=72.0,
            std_dev=36.0,
            min_val=24.0,
            max_val=240.0,
            distribution_type="lognormal"
        ),
        MTTR_PV=UncertaintyDistribution(
            name="PV_MTTR",
            mean=48.0,
            std_dev=24.0,
            min_val=12.0,
            max_val=144.0,
            distribution_type="lognormal"
        ),
        MTTR_DG=UncertaintyDistribution(
            name="DG_MTTR",
            mean=36.0,
            std_dev=18.0,
            min_val=12.0,
            max_val=120.0,
            distribution_type="lognormal"
        ),
        MTTR_BAT=UncertaintyDistribution(
            name="BAT_MTTR",
            mean=24.0,
            std_dev=12.0,
            min_val=6.0,
            max_val=72.0,
            distribution_type="lognormal"
        ),
        
        # 燃料不確定性
        fuel_storage_uncertainty=UncertaintyDistribution(
            name="fuel_storage_capacity",
            mean=1.0,
            std_dev=0.15,  # ±15%
            min_val=0.70,
            max_val=1.30,
            distribution_type="normal"
        ),
        fuel_consumption_rate_uncertainty=UncertaintyDistribution(
            name="fuel_consumption_rate",
            mean=1.0,
            std_dev=0.20,  # ±20%
            min_val=0.60,
            max_val=1.50,
            distribution_type="normal"
        ),
        
        num_simulations=num_simulations,
        random_seed=42,  # 可重現性
    )
    
    print(f"\n配置參數:")
    print(f"  模擬次數: {num_simulations}")
    print(f"  故障率 (平均值):")
    print(f"    - WT: {mc_config.p_fail_WT.mean:.1%}")
    print(f"    - PV: {mc_config.p_fail_PV.mean:.1%}")
    print(f"    - DG: {mc_config.p_fail_DG.mean:.1%}")
    print(f"    - BAT: {mc_config.p_fail_BAT.mean:.1%}")
    print(f"  修復時間 (平均值):")
    print(f"    - WT: {mc_config.MTTR_WT.mean:.1f} 小時")
    print(f"    - PV: {mc_config.MTTR_PV.mean:.1f} 小時")
    print(f"    - DG: {mc_config.MTTR_DG.mean:.1f} 小時")
    print(f"    - BAT: {mc_config.MTTR_BAT.mean:.1f} 小時")
    print(f"  燃料不確定性:")
    print(f"    - 儲存容量: ±{mc_config.fuel_storage_uncertainty.std_dev:.0%}")
    print(f"    - 消耗率: ±{mc_config.fuel_consumption_rate_uncertainty.std_dev:.0%}")
    print()
    
    # 執行分析
    analyzer = MonteCarloAnalyzer(config=mc_config)
    mc_result = analyzer.run_analysis(
        design=design,
        scenario=scenario,
        time_input=time_input,
        critical_load_ratio=0.2,
        ems_policy=ems_policy,
    )
    
    # 生成報告
    os.makedirs("charts/monte_carlo", exist_ok=True)
    report_text = analyzer.generate_report(
        mc_result,
        output_path="charts/monte_carlo/MC_Uncertainty_Analysis_Report.txt"
    )
    
    return mc_result, analyzer


def save_monte_carlo_results(mc_result, analyzer):
    """保存 Monte Carlo 結果"""
    os.makedirs("charts/monte_carlo", exist_ok=True)
    
    # 保存統計摘要
    summary_df = mc_result.to_dataframe()
    summary_df.to_csv("charts/monte_carlo/MC_Summary_Statistics.csv", index=False)
    
    # 保存所有樣本
    samples_df = pd.DataFrame({
        'NPR': mc_result.NPR_samples,
        'CLSR': mc_result.CLSR_samples,
        'EID': mc_result.EID_samples,
        'Fuel_Consumed_gal': mc_result.fuel_consumed_samples,
        'Fuel_Remaining_gal': mc_result.fuel_remaining_samples,
    })
    samples_df.to_csv("charts/monte_carlo/MC_All_Samples.csv", index=False)
    
    # 保存故障統計
    failure_stats_data = []
    for equipment, stats in mc_result.equipment_failure_stats.items():
        failure_stats_data.append({
            'Equipment': equipment,
            'Mean_Failures': stats['mean_failures_per_sim'],
            'Std_Failures': stats['std_failures'],
            'Max_Failures': stats['max_failures'],
            'Failure_Probability': stats['failure_probability'],
        })
    failure_df = pd.DataFrame(failure_stats_data)
    failure_df.to_csv("charts/monte_carlo/MC_Equipment_Failure_Stats.csv", index=False)
    
    # 保存修復時間統計
    repair_stats_data = []
    for equipment, stats in mc_result.repair_time_stats.items():
        repair_stats_data.append({
            'Equipment': equipment,
            'Mean_Hours': stats['mean'],
            'Std_Hours': stats['std'],
            'Min_Hours': stats['min'],
            'Median_Hours': stats['median'],
            'Max_Hours': stats['max'],
        })
    repair_df = pd.DataFrame(repair_stats_data)
    repair_df.to_csv("charts/monte_carlo/MC_Repair_Time_Stats.csv", index=False)
    
    print("\n【結果文件已保存】")
    print("  - MC_Summary_Statistics.csv")
    print("  - MC_All_Samples.csv")
    print("  - MC_Equipment_Failure_Stats.csv")
    print("  - MC_Repair_Time_Stats.csv")
    print("  - MC_Uncertainty_Analysis_Report.txt")


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)
    
    print("\n" + "=" * 80)
    print("微電網韌性模擬 + Monte Carlo 不確定分析")
    print("=" * 80)
    
    # ============================================================
    # 1. 讀取氣象和負載資料
    # ============================================================
    print("\n【步驟 1】讀取氣象和負載資料...")
    
    df_weather = pd.read_csv("Morakot_weather_with_demand.csv")
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    df_weather["timestamp"] = df_weather["datetime"]
    
    demand_series = df_weather["demand"].tolist()
    N = len(df_weather)
    days_in_dataset = N / 24.0
    
    print(f"  已載入 {N} 小時的資料（{days_in_dataset:.1f} 天）")
    
    # ============================================================
    # 2. 建立時間序列和危險值
    # ============================================================
    print("【步驟 2】建立時間序列和危險值...")
    
    time_input = build_time_input(df_weather, demand_series)
    hazard = build_hazard_from_weather(df_weather)
    
    # ============================================================
    # 3. 定義災害情景
    # ============================================================
    print("【步驟 3】定義災害情景...")
    
    td_ts = pd.Timestamp("2009-08-07 00:00:00")
    tfr_ts = pd.Timestamp("2009-08-10 00:00:00")
    
    try:
        disturbance_start = df_weather.index[df_weather["datetime"] == td_ts][0]
        disturbance_end = df_weather.index[df_weather["datetime"] == tfr_ts][0]
    except IndexError:
        disturbance_start = (df_weather["datetime"] - td_ts).abs().idxmin()
        disturbance_end = (df_weather["datetime"] - tfr_ts).abs().idxmin()
    
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
    
    print(f"  災害時間: {disturbance_start} ~ {disturbance_end}")
    
    # ============================================================
    # 4. 定義微電網設計
    # ============================================================
    print("【步驟 4】定義微電網設計...")
    
    design_baseline = MicrogridDesign(
        P_WT=[300.0],
        P_PV=[200.0],
        P_DG=[500.0],
        P_BAT=[100.0],
        B_max=[400.0],
        B_init=[200.0],
        eta_c=0.95,
        eta_d=0.95,
        C_WT=0.35,
        C_PV=0.20,
        A_WT=1000.0,
        fuel_rate_max=50.0,
        fuel_storage=compute_fuel_storage(50.0, 5, [500.0]),
        DG_min_loading=0.2,
        DG_max_loading=1.0,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=0.5,
        C_rate_discharge=0.5,
    )
    
    design_enhanced = MicrogridDesign(
        P_WT=[400.0],
        P_PV=[300.0],
        P_DG=[500.0],
        P_BAT=[150.0],
        B_max=[500.0],
        B_init=[250.0],
        eta_c=0.95,
        eta_d=0.95,
        C_WT=0.35,
        C_PV=0.20,
        A_WT=1000.0,
        fuel_rate_max=50.0,
        fuel_storage=compute_fuel_storage(50.0, 7, [500.0]),
        DG_min_loading=0.2,
        DG_max_loading=1.0,
        B_min_soc_frac=0.2,
        B_max_soc_frac=0.8,
        C_rate_charge=0.5,
        C_rate_discharge=0.5,
    )
    
    designs = [design_baseline, design_enhanced]
    print(f"  已定義 {len(designs)} 個設計方案")
    
    # ============================================================
    # 5. 定義成本參數
    # ============================================================
    print("【步驟 5】定義成本參數...")
    
    cost_params = CostParameters(
        I_WT=[500000],
        I_PV=[300000],
        I_DG=[200000],
        I_BAT=[50000],
        M_WT=[15000],
        M_PV=[9000],
        M_DG=[12000],
        M_BAT=[2000],
        H_WT=[100000],
        H_PV=[30000],
        H_DG=[50000],
        H_BAT=[10000],
        VOLL=50.0,
        C_fix_baseline=10000.0,
        C_fix_strategy=15000.0,
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
    # 6. 執行確定性模擬
    # ============================================================
    results, rbcr_results = run_deterministic_simulation(
        df_weather,
        time_input,
        hazard,
        scenario_hurricane,
        designs,
        cost_params,
        ems_policy,
    )
    
    # ============================================================
    # 7. 執行 Monte Carlo 分析（基準設計）
    # ============================================================
    mc_result, analyzer = run_monte_carlo_analysis(
        design=design_baseline,
        scenario=scenario_hurricane,
        time_input=time_input,
        ems_policy=ems_policy,
        num_simulations=1000,
    )
    
    # ============================================================
    # 8. 保存 Monte Carlo 結果
    # ============================================================
    save_monte_carlo_results(mc_result, analyzer)
    
    # ============================================================
    # 9. 繪製可視化
    # ============================================================
    print("\n【步驟 9】繪製可視化圖表...")
    
    # 繪製 Monte Carlo 結果
    plot_monte_carlo_results(mc_result, output_dir="charts/monte_carlo")
    
    # 執行一次確定性模擬用於繪製詳細圖表
    from microgrid.simulation import simulate_microgrid_resilience
    sim_result = simulate_microgrid_resilience(
        design=design_baseline,
        scenario=scenario_hurricane,
        time_input=time_input,
        critical_load_ratio=0.2,
        random_seed=42,
        ems_policy=ems_policy,
    )
    
    plot_all_figures(
        df=df_weather,
        sim_result=sim_result,
        scenario=scenario_hurricane,
        disturbance_start=disturbance_start,
        disturbance_end=disturbance_end,
        output_dir="charts",
    )

    plot_ems_timeseries(
        df=df_weather,
        sim_result=sim_result,
        scenario=scenario_hurricane,
        disturbance_start=disturbance_start,
        disturbance_end=disturbance_end,
        output_dir="charts",
    )
    
    # ============================================================
    # 10. 完成
    # ============================================================
    print("\n" + "=" * 80)
    print("模擬和分析完成！")
    print("=" * 80)
    print("\n【輸出位置】")
    print("  - charts/: 確定性模擬圖表")
    print("  - charts/monte_carlo/: Monte Carlo 分析結果")
    print("\n【主要輸出文件】")
    print("  - MC_Uncertainty_Analysis_Report.txt")
    print("  - MC_Summary_Statistics.csv")
    print("  - MC_All_Samples.csv")
    print("  - MC_Equipment_Failure_Stats.csv")
    print("  - MC_Repair_Time_Stats.csv")
    print("  - *.png 圖表檔案")
