# microgrid/plotting.py
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
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
