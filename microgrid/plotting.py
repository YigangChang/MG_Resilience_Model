# microgrid/plotting.py
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from .models import SimulationResult, DisturbanceScenario


def _plot_timeline(ax, data, title, start_idx, end_idx,
                   color_ok="green", color_bad="red"):
    ax.set_title(title)
    t = np.arange(start_idx, end_idx + 1)
    for unit_idx, series in enumerate(data):
        status = np.array(series[start_idx:end_idx + 1])

        ax.plot(
            t[status == 1],
            [unit_idx] * np.sum(status == 1),
            "|",
            color=color_ok,
            markersize=8,
        )
        ax.plot(
            t[status == 0],
            [unit_idx] * np.sum(status == 0),
            "|",
            color=color_bad,
            markersize=8,
        )
    ax.set_yticks(range(len(data)))
    ax.set_ylabel("Unit index")
    ax.set_xlabel("Hour index")


def plot_all_figures(
    df: pd.DataFrame,
    sim_result: SimulationResult,
    scenario: DisturbanceScenario,
    disturbance_start: int,
    disturbance_end: int,
    output_dir: str = "charts",
):
    """
    把原本 main 裡的所有 plt.* 繪圖程式搬進來，整合成一個函式。
    """
    os.makedirs(output_dir, exist_ok=True)

    T_len = len(sim_result.demand)
    idx_start = max(0, disturbance_start - 24)
    idx_end = min(T_len - 1, disturbance_start + scenario.evaluation_horizon_hours)

    timestamp_window = df["timestamp"].iloc[idx_start:idx_end + 1]
    demand_window = np.array(sim_result.demand)[idx_start:idx_end + 1]
    Gt_window = np.array(sim_result.Gt)[idx_start:idx_end + 1]
    Tt_window = np.array(sim_result.Tt)[idx_start:idx_end + 1]
    service_window = np.array(sim_result.service_level)[idx_start:idx_end + 1]

    B_array = np.array(sim_result.B)
    B_avg_window = B_array.mean(axis=0)[idx_start:idx_end + 1]

    P_wt_window = np.array(sim_result.P_wt)[idx_start:idx_end + 1]
    P_pv_window = np.array(sim_result.P_pv)[idx_start:idx_end + 1]
    P_dg_window = np.array(sim_result.P_dg)[idx_start:idx_end + 1]
    P_charge_total_window = np.sum(np.array(sim_result.P_charge)[:, idx_start:idx_end + 1], axis=0)
    P_discharge_total_window = np.sum(np.array(sim_result.P_discharge)[:, idx_start:idx_end + 1], axis=0)

    # ===== 圖 1：Demand / Served / Unserved =====
    plt.figure(figsize=(12, 5))
    plt.plot(timestamp_window, demand_window, label="Effective Demand [kW]")
    plt.plot(timestamp_window, Gt_window, label="Served Load Gt [kW]")
    plt.plot(timestamp_window, Tt_window, label="Unserved Load Tt [kW]")
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--", label="Disturbance start")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--", label="Disturbance end")
    plt.xlabel("Time")
    plt.ylabel("Power / kW")
    plt.title("Hourly Effective Demand, Served Load, and Unserved Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hourly_Demand_Served_Unserved.png"), dpi=200)
    plt.show()

    # ===== 圖 2：Battery SOC 平均 =====
    plt.figure(figsize=(12, 4))
    plt.plot(timestamp_window, B_avg_window)
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Battery SOC (average) [kWh]")
    plt.title("Hourly Battery SOC (Average)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hourly_Battery_SOC_Average.png"), dpi=200)
    plt.show()

    # ===== 圖 3：Resilience Curve =====
    plt.figure(figsize=(12, 4))
    plt.plot(timestamp_window, service_window, label="Service level")
    plt.axhline(0.95, linestyle="--", label="0.95 threshold")
    plt.axhline(0.99, linestyle="--", label="0.99 threshold")
    plt.axvspan(df["timestamp"].iloc[disturbance_start], df["timestamp"].iloc[disturbance_end], alpha=0.2)
    plt.ylim(0, 1.05)
    plt.xlabel("Time")
    plt.ylabel("Service level [-]")
    plt.title("Resilience Curve (Service Level Over Time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Resilience_Curve_Service_Level.png"), dpi=200)
    plt.show()

    # ===== 圖 4：各機組發電 + 電池充放電 =====~~~
    plt.figure(figsize=(12, 5))
    plt.plot(timestamp_window, P_wt_window, label="Wind generation [kW]")
    plt.plot(timestamp_window, P_pv_window, label="Solar generation [kW]")
    plt.plot(timestamp_window, P_dg_window, label="Diesel generation [kW]")
    plt.plot(timestamp_window, P_charge_total_window, label="Battery charge (total) [kW]")
    plt.plot(timestamp_window, P_discharge_total_window, label="Battery discharge (total) [kW]")
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--", label="Disturbance start")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--", label="Disturbance end")
    plt.xlabel("Time")
    plt.ylabel("Power [kW]")
    plt.title("Hourly Generation and Battery Charge/Discharge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hourly_Generation_and_Battery.png"), dpi=200)
    plt.show()

    # ===== 圖 5：DER Availability Timeline (WT, DG, BAT) =====
    start_idx = max(0, disturbance_start - 24)
    end_idx = min(T_len - 1, disturbance_end + 168)

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    _plot_timeline(ax1, sim_result.U_WT_series, "WT Availability Timeline", start_idx, end_idx)

    ax2 = plt.subplot(3, 1, 2)
    _plot_timeline(ax2, sim_result.U_DG_series, "DG Availability Timeline", start_idx, end_idx)

    ax3 = plt.subplot(3, 1, 3)
    _plot_timeline(ax3, sim_result.U_BAT_series, "Battery Availability Timeline", start_idx, end_idx)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DER_Availability_Timeline_WT_DG_BAT.png"), dpi=200)
    plt.show()

    # ===== PV 獨立一張 =====
    plt.figure(figsize=(14, 8))
    _plot_timeline(
        plt.gca(),
        sim_result.U_PV_series,
        "PV Availability Timeline (Isolated)",
        start_idx,
        end_idx,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PV_Availability_Timeline.png"), dpi=200)
    plt.show()

    print("完成圖表繪製!")


def plot_ems_timeseries(
    df: pd.DataFrame,
    sim_result: SimulationResult,
    scenario: DisturbanceScenario,
    disturbance_start: int,
    disturbance_end: int,
    output_dir: str = "charts",
):
    """
    Plot EMS-related time series (avg SOC, load tier ratio, EMS mode).
    """
    os.makedirs(output_dir, exist_ok=True)

    T_len = len(sim_result.demand)
    idx_start = max(0, disturbance_start - 24)
    idx_end = min(T_len - 1, disturbance_start + scenario.evaluation_horizon_hours)

    timestamp_window = df["timestamp"].iloc[idx_start:idx_end + 1]
    soc_window = np.array(sim_result.avg_soc_frac)[idx_start:idx_end + 1]
    tier_window = np.array(sim_result.load_tier_ratio)[idx_start:idx_end + 1]
    mode_window = np.array(sim_result.ems_mode)[idx_start:idx_end + 1]

    mode_map = {
        "pre_event_charge": 0,
        "island_tier_1": 1,
        "island_tier_2": 2,
        "island_tier_3": 3,
        "grid_ems_charge": 4,
        "grid_ems_discharge": 5,
        "grid_ems_normal": 6,
        "normal": 7,
    }
    mode_values = np.array([mode_map.get(m, 8) for m in mode_window])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(timestamp_window, soc_window, label="Avg SOC")
    axes[0].axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    axes[0].axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    axes[0].set_ylabel("SOC fraction")
    axes[0].set_title("EMS: Average SOC")
    axes[0].legend()

    axes[1].plot(timestamp_window, tier_window, color="tab:orange", label="Load tier ratio")
    axes[1].axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    axes[1].axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    axes[1].set_ylabel("Tier ratio")
    axes[1].set_title("EMS: Load Tier Ratio")
    axes[1].legend()

    axes[2].step(timestamp_window, mode_values, where="post", color="tab:green")
    axes[2].axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    axes[2].axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    axes[2].set_ylabel("Mode index")
    axes[2].set_title("EMS: Mode Timeline")
    axes[2].set_yticks(sorted(set(mode_values)))
    mode_labels = {}
    for k, v in mode_map.items():
        mode_labels[v] = k
    axes[2].set_yticklabels([mode_labels.get(v, "other") for v in axes[2].get_yticks()])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "EMS_TimeSeries.png"), dpi=200)
    plt.show()


def plot_monte_carlo_results(
    mc_result,
    output_dir: str = "charts/monte_carlo",
):
    """
    繪製 Monte Carlo 分析結果的可視化圖表
    
    Parameters:
    -----------
    mc_result : MonteCarloResult
        Monte Carlo 分析結果
    output_dir : str
        輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 圖 1：韌性指標分佈（直方圖） =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(mc_result.NPR_samples, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(mc_result.NPR_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mc_result.NPR_mean:.4f}')
    axes[0].axvline(mc_result.NPR_median, color='green', linestyle='--', linewidth=2, label=f'Median: {mc_result.NPR_median:.4f}')
    axes[0].set_xlabel('NPR')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Nadir Performance Ratio (NPR) Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(mc_result.CLSR_samples, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(mc_result.CLSR_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mc_result.CLSR_mean:.4f}')
    axes[1].axvline(mc_result.CLSR_median, color='orange', linestyle='--', linewidth=2, label=f'Median: {mc_result.CLSR_median:.4f}')
    axes[1].set_xlabel('CLSR')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Critical Load Survival Ratio (CLSR) Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(mc_result.EID_samples, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[2].axvline(mc_result.EID_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mc_result.EID_mean:.4f}')
    axes[2].axvline(mc_result.EID_median, color='purple', linestyle='--', linewidth=2, label=f'Median: {mc_result.EID_median:.4f}')
    axes[2].set_xlabel('EID')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Energy Insufficiency Duration (EID) Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_Resilience_Indicators_Distribution.png'), dpi=200)
    plt.close()
    
    # ===== 圖 2：百分位數區間 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    indicators = ['NPR', 'CLSR', 'EID']
    means = [mc_result.NPR_mean, mc_result.CLSR_mean, mc_result.EID_mean]
    stds = [mc_result.NPR_std, mc_result.CLSR_std, mc_result.EID_std]
    p5s = [mc_result.NPR_percentile_5, mc_result.CLSR_percentile_5, mc_result.EID_percentile_5]
    p95s = [mc_result.NPR_percentile_95, mc_result.CLSR_percentile_95, mc_result.EID_percentile_95]
    
    for idx, ax in enumerate(axes):
        ax.bar(['Mean', '5%ile', '95%ile'], 
               [means[idx], p5s[idx], p95s[idx]],
               color=['red', 'blue', 'green'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title(f'{indicators[idx]} Statistics')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_Percentile_Statistics.png'), dpi=200)
    plt.close()
    
    # ===== 圖 3：燃料消耗分佈 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(mc_result.fuel_consumed_samples, bins=50, color='brown', alpha=0.7, edgecolor='black')
    axes[0].axvline(mc_result.fuel_consumed_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mc_result.fuel_consumed_mean:.2f} gal')
    axes[0].set_xlabel('Fuel Consumed (gallons)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Diesel Fuel Consumption Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(mc_result.fuel_remaining_samples, bins=50, color='teal', alpha=0.7, edgecolor='black')
    axes[1].axvline(mc_result.fuel_remaining_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mc_result.fuel_remaining_mean:.2f} gal')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=2, label='zero fuel remaining')
    axes[1].set_xlabel('Fuel Remaining (gallons)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Fuel Remaining Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_Fuel_Distribution.png'), dpi=200)
    plt.close()
    
    # ===== 圖 4：設備故障統計 =====
    equipment_names = {'WT': 'Wind Turbine', 'PV': 'Photovoltaic', 'DG': 'Diesel Generator', 'BAT': 'Battery'}
    equipment_keys = ['WT', 'PV', 'DG', 'BAT']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, eq_key in enumerate(equipment_keys):
        stats = mc_result.equipment_failure_stats[eq_key]
        ax = axes[idx]
        
        categories = ['Average Failures', 'Failure Probability']
        values = [stats['mean_failures_per_sim'], stats['failure_probability'] * 100]
        colors = ['skyblue', 'salmon']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title(f'{equipment_names.get(eq_key, eq_key)} Failure Statistics')
        ax.grid(alpha=0.3, axis='y')
        
        # 在柱狀圖上添加數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_Equipment_Failure_Statistics.png'), dpi=200)
    plt.close()
    
    # ===== 圖 5：修復時間分佈 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, eq_key in enumerate(equipment_keys):
        repair_times = mc_result.repair_time_stats[eq_key]
        ax = axes[idx]
        
        times = [repair_times['min'], repair_times['median'], repair_times['max']]
        labels = ['Min', 'Median', 'Max']
        colors = ['lightblue', 'gold', 'lightcoral']
        
        bars = ax.bar(labels, times, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Repair Time (hours)')
        ax.set_title(f'{equipment_names.get(eq_key, eq_key)} Repair Time Statistics')
        ax.grid(alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, value in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}h',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_Repair_Time_Statistics.png'), dpi=200)
    plt.close()
    
    # ===== 圖 6：累積分佈函數 (CDF) =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # NPR CDF
    sorted_npr = np.sort(mc_result.NPR_samples)
    cdf_npr = np.arange(1, len(sorted_npr) + 1) / len(sorted_npr)
    axes[0].plot(sorted_npr, cdf_npr, linewidth=2, color='blue')
    axes[0].axvline(mc_result.NPR_percentile_5, color='green', linestyle='--', label='5%ile')
    axes[0].axvline(mc_result.NPR_percentile_95, color='red', linestyle='--', label='95%ile')
    axes[0].set_xlabel('NPR')
    axes[0].set_ylabel('accumulative probability')
    axes[0].set_title('NPR Cumulative Distribution Function')
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # CLSR CDF
    sorted_clsr = np.sort(mc_result.CLSR_samples)
    cdf_clsr = np.arange(1, len(sorted_clsr) + 1) / len(sorted_clsr)
    axes[1].plot(sorted_clsr, cdf_clsr, linewidth=2, color='green')
    axes[1].axvline(mc_result.CLSR_percentile_5, color='blue', linestyle='--', label='5%ile')
    axes[1].axvline(mc_result.CLSR_percentile_95, color='orange', linestyle='--', label='95%ile')
    axes[1].set_xlabel('CLSR')
    axes[1].set_ylabel('accumulative probability')
    axes[1].set_title('CLSR Cumulative Distribution Function')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # EID CDF
    sorted_eid = np.sort(mc_result.EID_samples)
    cdf_eid = np.arange(1, len(sorted_eid) + 1) / len(sorted_eid)
    axes[2].plot(sorted_eid, cdf_eid, linewidth=2, color='orange')
    axes[2].axvline(mc_result.EID_percentile_5, color='purple', linestyle='--', label='5%ile')
    axes[2].axvline(mc_result.EID_percentile_95, color='red', linestyle='--', label='95%ile')
    axes[2].set_xlabel('EID')
    axes[2].set_ylabel('accumulative probability')
    axes[2].set_title('EID Cumulative Distribution Function')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MC_CDF_Curves.png'), dpi=200)
    plt.close()
    
    print(f"Monte Carlo 分析圖表已保存至: {output_dir}")
