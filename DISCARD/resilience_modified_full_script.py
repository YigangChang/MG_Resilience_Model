"""
Microgrid resilience simulation with:
- Normal mode (grid available, full demand)
- Pre-storm preparation (24h full battery charge to 90% SOC)
- Disaster mode (grid outage, critical load only, WT/PV/DG/BESS supply)
- Hazard-based failures
- IEEE-style resiliency curve
- LCOE / LCOED with diesel fuel cost tied to simulation

Requires:
- Microgrid_5to11_months_timeseries.csv
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# =========================================================
# 1. 資料結構定義
# =========================================================

@dataclass
class MicrogridDesign:
    """微電網設備配置 + 運轉限制"""
    P_WT: List[float]
    P_PV: List[float]
    P_DG: List[float]
    P_BAT: List[float]

    B_max: List[float]
    B_init: List[float]

    eta_c: float
    eta_d: float

    C_WT: float
    C_PV: float
    A_WT: float

    fuel_rate_max: float
    fuel_storage: float

    DG_min_loading: float
    DG_max_loading: float

    B_min_soc_frac: float   # 非災害一般下限（例如 0.2）
    B_max_soc_frac: float   # 非災害一般上限（例如 0.8）
    C_rate_charge: float
    C_rate_discharge: float


@dataclass
class HazardProfile:
    time_h: List[int]
    wind_speed: List[float]        # m/s
    rainfall: List[float]          # mm/hr
    solar_irradiance: List[float]  # 0~1


@dataclass
class DisturbanceScenario:
    name: str
    disturbance_start: int
    disturbance_end: int
    base_p_damage_WT: float
    base_p_damage_PV: float
    base_p_damage_DG: float
    base_p_damage_BAT: float
    MTTR_WT: float
    MTTR_PV: float
    MTTR_DG: float
    MTTR_BAT: float
    hazard: HazardProfile
    evaluation_horizon_hours: int = 168  # Resiliency Curve 評估窗（預設 7 天）


@dataclass
class TimeSeriesInput:
    demand: List[float]
    cf_WT: List[float]
    cf_PV: List[float]
    hours: Optional[List[int]] = None  # 0~23


@dataclass
class SimulationResult:
    invulnerability: float
    resilience_curve: float
    recovery_time_h: Optional[float]

    Pt: List[float]
    Gt: List[float]
    Tt: List[float]
    demand: List[float]  # 這裡用「有效 demand」（災害期間是 critical load）

    B: List[List[float]]
    P_charge: List[List[float]]
    P_discharge: List[List[float]]
    curtailment: List[float]

    fuel_used: float
    service_level: List[float]

    P_wt: List[float]
    P_pv: List[float]
    P_dg: List[float]

    U_WT_series: List[List[int]]
    U_PV_series: List[List[int]]
    U_DG_series: List[List[int]]
    U_BAT_series: List[List[int]]


# =========================================================
# 2. 核心韌性模擬 simulate_microgrid_resilience
# =========================================================

def simulate_microgrid_resilience(
    design: MicrogridDesign,
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    critical_load_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> SimulationResult:
    """
    新版微電網韌性模擬：

    非災害期間（Normal mode, grid available）:
      - 負載 = 100% demand
      - 儲能 SOC 目標: 20%~80%
      - 儲能充電來源：WT/PV 優先，有多餘才充；若在災前 24h，視為用主網補到 90%
      - 替用供電順序：WT/PV + (可選 BESS 放電) + Grid
      - 使用者角度無缺電: service_level = 1

    災害前 24 小時（Pre-storm mode, grid available）:
      - 同正常模式，但強制將 BESS SOC 拉到 90%（設計上限可到 90%）
      - 等效為「主網幫忙充飽」。

    災害期間（Island mode, grid unavailable）:
      - 主網停電，完全不能用 grid
      - 負載 = critical_load_ratio * demand（例如 20%）
      - 儲能 SOC 範圍: 10%~90%
      - 供電順序：WT/PV → DG (min/max loading) → BESS → Unserved
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # 完整 demand 與容量因子
    D_full = time_input.demand
    cf_WT_ts = time_input.cf_WT
    cf_PV_ts = time_input.cf_PV
    T_len = len(D_full)

    n_WT = len(design.P_WT)
    n_PV = len(design.P_PV)
    n_DG = len(design.P_DG)
    n_BAT = len(design.P_BAT)

    # 初始設備可用狀態
    U_WT = [1] * n_WT
    U_PV = [1] * n_PV
    U_DG = [1] * n_DG
    U_BAT = [1] * n_BAT

    U_WT_series = [[1] * T_len for _ in range(n_WT)]
    U_PV_series = [[1] * T_len for _ in range(n_PV)]
    U_DG_series = [[1] * T_len for _ in range(n_DG)]
    U_BAT_series = [[1] * T_len for _ in range(n_BAT)]

    repair_WT = [None] * n_WT
    repair_PV = [None] * n_PV
    repair_DG = [None] * n_DG
    repair_BAT = [None] * n_BAT

    # 電池 SOC
    B = [[0.0] * T_len for _ in range(n_BAT)]
    for i in range(n_BAT):
        B[i][0] = design.B_init[i]

    P_charge = [[0.0] * T_len for _ in range(n_BAT)]
    P_discharge = [[0.0] * T_len for _ in range(n_BAT)]
    curtailment = [0.0] * T_len

    Pt = [0.0] * T_len
    Gt = [0.0] * T_len
    Tt = [0.0] * T_len
    service_level = [1.0] * T_len

    # 有效 demand：平時 = 全部負載；災害期間 = 關鍵負載
    D_effective = [0.0] * T_len

    # 每小時風、光、柴油輸出
    P_wt_all = [0.0] * T_len
    P_pv_all = [0.0] * T_len
    P_dg_all = [0.0] * T_len

    fuel_used = 0.0
    fuel_remaining = design.fuel_storage

    td = scenario.disturbance_start
    tfr = min(scenario.disturbance_end, T_len - 1)
    t_eval_end = min(T_len - 1, td + scenario.evaluation_horizon_hours)

    # 溫度/應力降額：目前只用在電池 C-rate 上
    temp_derate = np.ones(T_len, dtype=float)
    for t in range(T_len):
        h_wind = scenario.hazard.wind_speed[t]
        h_rain = scenario.hazard.rainfall[t]
        # 災害強度越高 -> C-rate 降額越大（最多 20%）
        hazard_factor = min(1.0, (h_wind / 25.0) + (h_rain / 80.0))
        temp_derate[t] = 1.0 - 0.2 * hazard_factor

    # 災後 24 小時 PV / WT 線性恢復
    derate_PV = np.ones(T_len, dtype=float)
    derate_WT = np.ones(T_len, dtype=float)
    post_storm_window = 24
    for t in range(T_len):
        if tfr < t <= tfr + post_storm_window:
            alpha = (t - tfr) / post_storm_window
            factor = 0.6 + 0.4 * alpha  # 從 60% 線性恢復到 100%
            derate_PV[t] = factor
            derate_WT[t] = factor

    # ========= 逐時模擬 =========
    for t in range(T_len):
        Dt_full = D_full[t]

        # 判斷模式
        in_storm = (td <= t <= tfr)
        pre_storm_window = (t < td) and ((td - t) <= 24)

        if in_storm:
            # 災害期間：只需供應 critical load
            Dt = Dt_full * critical_load_ratio
            grid_available = False
            # 災害期間 SOC 允許 10%~90%
            soc_min_frac = 0.10
            soc_max_frac = 0.90
        else:
            # 非災害期間：供應全負載，grid 可用
            Dt = Dt_full
            grid_available = True
            # 平時 SOC 限制：20%~80%，但災前 24 小時允許充到 90%
            soc_min_frac = design.B_min_soc_frac  # 一般 0.2
            soc_max_frac = design.B_max_soc_frac  # 一般 0.8
            if pre_storm_window:
                soc_max_frac = 0.90

        D_effective[t] = Dt

        # -------- 1) 故障 / 修復（只在災害期間考慮 hazard-based p_fail）--------
        if in_storm:
            h_wind = scenario.hazard.wind_speed[t]
            h_rain = scenario.hazard.rainfall[t]
            # 災害期間：用平方放大高風速 / 強降雨的影響
            hazard_factor = min(1.0, (h_wind / 25.0) ** 2 + (h_rain / 80.0) ** 2)
        else:
            hazard_factor = 0.0

        # WT
        for i in range(n_WT):
            if U_WT[i] == 1:
                p_fail = scenario.base_p_damage_WT * hazard_factor
                if random.random() < p_fail:
                    U_WT[i] = 0
                    repair_WT[i] = t + math.ceil(scenario.MTTR_WT)
            else:
                if repair_WT[i] is not None and t >= repair_WT[i]:
                    U_WT[i] = 1

        # PV
        for i in range(n_PV):
            if U_PV[i] == 1:
                p_fail = scenario.base_p_damage_PV * hazard_factor
                if random.random() < p_fail:
                    U_PV[i] = 0
                    repair_PV[i] = t + math.ceil(scenario.MTTR_PV)
            else:
                if repair_PV[i] is not None and t >= repair_PV[i]:
                    U_PV[i] = 1

        # DG
        for i in range(n_DG):
            if U_DG[i] == 1:
                p_fail = scenario.base_p_damage_DG * hazard_factor
                if random.random() < p_fail:
                    U_DG[i] = 0
                    repair_DG[i] = t + math.ceil(scenario.MTTR_DG)
            else:
                if repair_DG[i] is not None and t >= repair_DG[i]:
                    U_DG[i] = 1

        # BAT
        for i in range(n_BAT):
            if U_BAT[i] == 1:
                p_fail = scenario.base_p_damage_BAT * hazard_factor
                if random.random() < p_fail:
                    U_BAT[i] = 0
                    repair_BAT[i] = t + math.ceil(scenario.MTTR_BAT)
            else:
                if repair_BAT[i] is not None and t >= repair_BAT[i]:
                    U_BAT[i] = 1

        # 紀錄可用狀態
        for i in range(n_WT):
            U_WT_series[i][t] = U_WT[i]
        for i in range(n_PV):
            U_PV_series[i][t] = U_PV[i]
        for i in range(n_DG):
            U_DG_series[i][t] = U_DG[i]
        for i in range(n_BAT):
            U_BAT_series[i][t] = U_BAT[i]

        # -------- 2) SOC 傳遞與邊界 --------
        if t > 0:
            for i in range(n_BAT):
                B[i][t] = B[i][t - 1]

        # 根據當前模式決定 SOC 上下限
        for i in range(n_BAT):
            B_max_i = design.B_max[i]
            soc_min = soc_min_frac * B_max_i
            soc_max = soc_max_frac * B_max_i
            B[i][t] = min(max(B[i][t], soc_min), soc_max)

        # 災前 24 小時：預充電到 90%（視為主網充電，略過能量平衡）
        if pre_storm_window:
            for i in range(n_BAT):
                B[i][t] = max(B[i][t], 0.90 * design.B_max[i])

        # -------- 3) WT / PV 輸出 --------
        P_wt_t = 0.0
        for i in range(n_WT):
            if U_WT[i] == 1:
                P_wt_t += design.P_WT[i] * cf_WT_ts[t] * design.A_WT * derate_WT[t]

        P_pv_t = 0.0
        for i in range(n_PV):
            if U_PV[i] == 1:
                cf_pv_eff = cf_PV_ts[t]
                # 災害期間 PV 最多 5% 額定
                if in_storm:
                    cf_pv_eff = min(cf_pv_eff, 0.05)
                P_pv_t += design.P_PV[i] * cf_pv_eff * derate_PV[t]

        P_wt_all[t] = P_wt_t
        P_pv_all[t] = P_pv_t

        # -------- 4) Diesel dispatch (只在災害期間使用) --------
        P_dg_t = 0.0
        if in_storm and fuel_remaining > 0 and Dt > 0:
            dg_min = design.DG_min_loading
            dg_max = design.DG_max_loading
            residual_after_RE = max(0.0, Dt - (P_wt_t + P_pv_t))
            for i in range(n_DG):
                if U_DG[i] == 0:
                    continue
                if fuel_remaining <= 0:
                    break
                P_rated = design.P_DG[i]
                unit_max = P_rated * dg_max
                unit_min = P_rated * dg_min
                if residual_after_RE <= 0:
                    break
                desired = min(residual_after_RE, unit_max)
                if desired < unit_min:
                    # 若尚無 DG 上線，則以 min loading 開機
                    if P_dg_t == 0.0:
                        actual = unit_min
                    else:
                        continue
                else:
                    actual = desired
                fuel_need = design.fuel_rate_max * (actual / P_rated)
                if fuel_remaining < fuel_need:
                    ratio = fuel_remaining / fuel_need
                    actual *= ratio
                    fuel_need = fuel_remaining
                P_dg_t += actual
                fuel_used += fuel_need
                fuel_remaining -= fuel_need
                residual_after_RE -= actual
                if fuel_remaining <= 0:
                    break

        P_dg_all[t] = P_dg_t

        # -------- 5) 電池充放電邏輯（分 Grid / Island 兩種模式） --------
        temp_factor = temp_derate[t]

        # 重置當小時電池功率
        for i in range(n_BAT):
            P_charge[i][t] = 0.0
            P_discharge[i][t] = 0.0

        if in_storm:
            # ======= 災害模式（無主網）=======
            P_gen_without_batt = P_wt_t + P_pv_t + P_dg_t
            T_without_batt = Dt - P_gen_without_batt  # >0 缺電, <0 多餘

            if T_without_batt < 0:
                # 多餘能量 -> 充電
                surplus = -T_without_batt
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if surplus <= 0:
                        break
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_max = soc_max_frac * B_max_i
                    if B_prev >= soc_max:
                        continue
                    P_rating = design.P_BAT[i] * temp_factor
                    P_c_rate = B_max_i * design.C_rate_charge
                    P_cap_limit = (soc_max - B_prev) / max(design.eta_c, 1e-9)
                    P_max_charge = min(P_rating, P_c_rate, P_cap_limit)
                    charge_power = min(surplus, P_max_charge)
                    if charge_power <= 0:
                        continue
                    P_charge[i][t] = charge_power
                    B[i][t] = B_prev + charge_power * design.eta_c
                    surplus -= charge_power
                curtailment[t] = surplus
                Gt[t] = Dt
                Tt[t] = 0.0

            elif T_without_batt > 0:
                # 缺電 -> 放電
                deficit = T_without_batt
                supplied_by_batt = 0.0
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if deficit <= 0:
                        break
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_min = soc_min_frac * B_max_i
                    if B_prev <= soc_min:
                        continue
                    P_rating = design.P_BAT[i] * temp_factor
                    P_c_rate = B_max_i * design.C_rate_discharge
                    max_energy_out_ac = (B_prev - soc_min) * design.eta_d
                    P_energy_limit = max_energy_out_ac
                    P_max_discharge = min(P_rating, P_c_rate, P_energy_limit)
                    discharge_power = min(deficit, P_max_discharge)
                    if discharge_power <= 0:
                        continue
                    P_discharge[i][t] = discharge_power
                    supplied_by_batt += discharge_power
                    deficit -= discharge_power
                    B[i][t] = max(
                        soc_min,
                        B_prev - discharge_power / max(design.eta_d, 1e-9),
                    )
                Gt[t] = P_gen_without_batt + supplied_by_batt
                Tt[t] = max(0.0, Dt - Gt[t])

            else:
                # 剛好平衡，電池不動
                Gt[t] = Dt
                Tt[t] = 0.0
                curtailment[t] = 0.0

        else:
            # ======= 非災害模式（Grid 可用）=======
            # Grid 永遠可以補足不足，因此使用者不會缺電。
            # 這裡的 Gt 定義為「微電網提供的負載」，但 resilience 用 service_level，
            # 非災害期直接設為 1.0。
            P_gen_RE = P_wt_t + P_pv_t
            served_by_RE = min(P_gen_RE, Dt)
            residual_load = Dt - served_by_RE
            surplus_RE = max(0.0, P_gen_RE - served_by_RE)

            # 選擇性：電池放電補負載（降低對主網依賴）
            # 為了簡化，我們僅在殘餘負載很高時使用少量電池（可再調整策略）
            supplied_by_batt = 0.0
            if residual_load > 0:
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if residual_load <= 0:
                        break
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_min = soc_min_frac * B_max_i
                    if B_prev <= soc_min:
                        continue
                    P_rating = design.P_BAT[i] * temp_factor
                    P_c_rate = B_max_i * design.C_rate_discharge
                    max_energy_out_ac = (B_prev - soc_min) * design.eta_d
                    P_energy_limit = max_energy_out_ac
                    P_max_discharge = min(P_rating, P_c_rate, P_energy_limit)
                    discharge_power = min(residual_load, P_max_discharge)
                    if discharge_power <= 0:
                        continue
                    P_discharge[i][t] = discharge_power
                    supplied_by_batt += discharge_power
                    residual_load -= discharge_power
                    B[i][t] = max(
                        soc_min,
                        B_prev - discharge_power / max(design.eta_d, 1e-9),
                    )

            # 主網補足剩餘負載（不記成本）
            grid_supply = max(0.0, residual_load)

            # 再使用 RE 裡的 surplus 充電（平時只用 RE，不用 grid 充）
            if surplus_RE > 0:
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if surplus_RE <= 0:
                        break
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_max = soc_max_frac * B_max_i
                    if B_prev >= soc_max:
                        continue
                    P_rating = design.P_BAT[i] * temp_factor
                    P_c_rate = B_max_i * design.C_rate_charge
                    P_cap_limit = (soc_max - B_prev) / max(design.eta_c, 1e-9)
                    P_max_charge = min(P_rating, P_c_rate, P_cap_limit)
                    charge_power = min(surplus_RE, P_max_charge)
                    if charge_power <= 0:
                        continue
                    P_charge[i][t] = charge_power
                    B[i][t] = B_prev + charge_power * design.eta_c
                    surplus_RE -= charge_power
            curtailment[t] = surplus_RE

            # 使用者角度總供電（微電網 + 主網）= Dt（完全滿足）
            Gt[t] = Dt
            Tt[t] = 0.0

        Pt[t] = P_wt_t + P_pv_t + P_dg_t + sum(design.P_BAT)

        # 服務水準：非災害期 = 1.0；災害期依實際供電 / critical load
        if in_storm:
            if Dt > 0:
                service_level[t] = min(1.0, Gt[t] / Dt)
            else:
                service_level[t] = 1.0
        else:
            service_level[t] = 1.0

    # ====================================================
    # 6. 韌性指標：Invulnerability + Resiliency Curve
    # ====================================================
    P_td = Pt[td] if 0 <= td < T_len else 0.0
    P_ts = Pt[tfr] if 0 <= tfr < T_len else 0.0
    invulnerability = (P_ts / P_td) if P_td > 0 else 0.0

    num = 0.0
    den = 0.0
    for t in range(td, t_eval_end + 1):
        num += (1.0 - service_level[t])
        den += 1.0
    resilience_curve = 1.0 - num / den if den > 0 else 0.0

    recovery_time_h = None
    for t in range(tfr, t_eval_end + 1):
        if service_level[t] >= 0.99:
            if all(level >= 0.95 for level in service_level[t:t_eval_end + 1]):
                recovery_time_h = t - td
                break

    return SimulationResult(
        invulnerability=invulnerability,
        resilience_curve=resilience_curve,
        recovery_time_h=recovery_time_h,
        Pt=Pt,
        Gt=Gt,
        Tt=Tt,
        demand=D_effective,
        B=B,
        P_charge=P_charge,
        P_discharge=P_discharge,
        curtailment=curtailment,
        fuel_used=fuel_used,
        service_level=service_level,
        P_wt=P_wt_all,
        P_pv=P_pv_all,
        P_dg=P_dg_all,
        U_WT_series=U_WT_series,
        U_PV_series=U_PV_series,
        U_DG_series=U_DG_series,
        U_BAT_series=U_BAT_series,
    )


# =========================================================
# 3. 成本模組：LCOE / LCOED（fuel cost 綁定模擬結果）
# =========================================================

@dataclass
class CostParameters:
    I_WT: List[float]
    I_PV: List[float]
    I_DG: List[float]
    I_BAT: List[float]

    M_WT: List[float]
    M_PV: List[float]
    M_DG: List[float]
    M_BAT: List[float]

    H_WT: List[float]
    H_PV: List[float]
    H_DG: List[float]
    H_BAT: List[float]

    planning_horizon_years: int
    wacc: float


def _npv_series(values: List[float], r: float) -> float:
    return sum(v / ((1 + r) ** (y + 1)) for y, v in enumerate(values))


def compute_LCOE(
    cost: CostParameters,
    Ey_year: List[float],
    fuel_used_sim: float,
    days_simulated: float = 214.0,
    fuel_price_per_gal: float = 97.0,
) -> float:
    r = cost.wacc
    p = cost.planning_horizon_years

    I_total = (
        sum(cost.I_WT)
        + sum(cost.I_PV)
        + sum(cost.I_DG)
        + sum(cost.I_BAT)
    )
    H_total = (
        sum(cost.H_WT)
        + sum(cost.H_PV)
        + sum(cost.H_DG)
        + sum(cost.H_BAT)
    )

    def pad(lst: List[float]) -> List[float]:
        if len(lst) >= p:
            return lst[:p]
        if not lst:
            return [0.0] * p
        return lst + [lst[-1]] * (p - len(lst))

    M_WT_y = pad(cost.M_WT)
    M_PV_y = pad(cost.M_PV)
    M_DG_y = pad(cost.M_DG)
    M_BAT_y = pad(cost.M_BAT)

    scale_to_year = 365.0 / days_simulated
    annual_fuel_gal = fuel_used_sim * scale_to_year
    F_DG_year = annual_fuel_gal * fuel_price_per_gal

    annual_costs = []
    for y in range(p):
        invest = I_total if y == 0 else 0.0
        salvage = H_total if y == p - 1 else 0.0
        annual = (
            invest
            + M_WT_y[y] + M_PV_y[y] + M_DG_y[y] + M_BAT_y[y]
            + F_DG_year
            - salvage
        )
        annual_costs.append(annual)

    numerator = _npv_series(annual_costs, r)
    denominator = _npv_series(Ey_year, r)
    return numerator / denominator if denominator > 0 else float("inf")


def compute_LCOED(
    cost: CostParameters,
    Dy_year: List[float],
    fuel_used_sim: float,
    days_simulated: float = 214.0,
    fuel_price_per_gal: float = 97.0,
) -> float:
    r = cost.wacc
    p = cost.planning_horizon_years

    I_total = (
        sum(cost.I_WT)
        + sum(cost.I_PV)
        + sum(cost.I_DG)
        + sum(cost.I_BAT)
    )
    H_total = (
        sum(cost.H_WT)
        + sum(cost.H_PV)
        + sum(cost.H_DG)
        + sum(cost.H_BAT)
    )

    def pad(lst: List[float]) -> List[float]:
        if len(lst) >= p:
            return lst[:p]
        if not lst:
            return [0.0] * p
        return lst + [lst[-1]] * (p - len(lst))

    M_WT_y = pad(cost.M_WT)
    M_PV_y = pad(cost.M_PV)
    M_DG_y = pad(cost.M_DG)
    M_BAT_y = pad(cost.M_BAT)

    scale_to_year = 365.0 / days_simulated
    annual_fuel_gal = fuel_used_sim * scale_to_year
    F_DG_year = annual_fuel_gal * fuel_price_per_gal

    annual_costs = []
    for y in range(p):
        invest = I_total if y == 0 else 0.0
        salvage = H_total if y == p - 1 else 0.0
        annual = (
            invest
            + M_WT_y[y] + M_PV_y[y] + M_DG_y[y] + M_BAT_y[y]
            + F_DG_year
            - salvage
        )
        annual_costs.append(annual)

    numerator = _npv_series(annual_costs, r)
    denominator = _npv_series(Dy_year, r)
    return numerator / denominator if denominator > 0 else float("inf")


def evaluate_designs(
    designs: List[MicrogridDesign],
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    base_cost: CostParameters,
    days_in_dataset: float,
    critical_load_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> List[Dict]:
    results = []
    scale_to_year = 365.0 / days_in_dataset

    for idx, design in enumerate(designs):
        sim = simulate_microgrid_resilience(
            design=design,
            scenario=scenario,
            time_input=time_input,
            critical_load_ratio=critical_load_ratio,
            random_seed=(None if random_seed is None else random_seed + idx),
        )
        Ey_total = sum(sim.Gt)
        Dy_total = sum(sim.demand)
        Ey_year = [Ey_total * scale_to_year]
        Dy_year = [Dy_total * scale_to_year]

        LCOE_val = compute_LCOE(
            cost=base_cost,
            Ey_year=Ey_year,
            fuel_used_sim=sim.fuel_used,
            days_simulated=days_in_dataset,
            fuel_price_per_gal=97.0,
        )
        LCOED_val = compute_LCOED(
            cost=base_cost,
            Dy_year=Dy_year,
            fuel_used_sim=sim.fuel_used,
            days_simulated=days_in_dataset,
            fuel_price_per_gal=97.0,
        )

        results.append(
            {
                "design_index": idx,
                "invulnerability": sim.invulnerability,
                "resilience_curve": sim.resilience_curve,
                "recovery_time_h": sim.recovery_time_h,
                "LCOE": LCOE_val,
                "LCOED": LCOED_val,
                "fuel_used": sim.fuel_used,
            }
        )

    return results


# =========================================================
# 4. 主程式：讀取資料、建 scenario、模擬、繪圖
# =========================================================

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

    # 災害情境：故障機率採 hazard-based
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
    P_DG_units = [5000.0] * 1  # 1 台 5 MW

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

    # 成本參數（示意）
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

    # ====================================================
    # 5. 視覺化：Demand / Served / SOC / Resilience Curve / DER timeline
    # ====================================================
    T_len = len(sim_result.demand)
    idx_start = max(0, disturbance_start - 24)
    idx_end = min(T_len - 1, disturbance_start + scenario_hurricane.evaluation_horizon_hours)

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

    # 圖 1：Demand / Served / Unserved
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
    plt.savefig("charts/Hourly_Demand_Served_Unserved.png", dpi=200)
    plt.show()

    # 圖 2：電池 SOC 平均
    plt.figure(figsize=(12, 4))
    plt.plot(timestamp_window, B_avg_window)
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Battery SOC (average) [kWh]")
    plt.title("Hourly Battery SOC (Average)")
    plt.tight_layout()
    plt.savefig("charts/Hourly_Battery_SOC_Average.png", dpi=200)
    plt.show()

    # 圖 3：Resilience Curve
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
    plt.savefig("charts/Resilience_Curve_Service_Level.png", dpi=200)
    plt.show()

    # 圖 4：各機組發電 + 電池充放電
    plt.figure(figsize=(12, 5))
    plt.plot(timestamp_window, P_wt_window, label="Wind generation [kW]")
    plt.plot(timestamp_window, P_pv_window, label="Solar generation [kW]")
    plt.plot(timestamp_window, P_dg_window, label="Diesel generation [kW]")
    plt.plot(timestamp_window, P_charge_total_window, label="Battery charge (total) [kW]")
    plt.plot(timestamp_window, P_discharge_total_window, label="Battery discharge (total) [kW]")
    plt.xlabel("Time")
    plt.ylabel("Power [kW]")
    plt.title("Hourly Generation and Battery Charge/Discharge")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/Hourly_Generation_and_Battery.png", dpi=200)
    plt.show()

    # 圖 5：DER Availability Timeline
    def plot_timeline(ax, data, title, start_idx, end_idx, color_ok="green", color_bad="red"):
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

    start_idx = max(0, disturbance_start - 24)
    end_idx = min(T_len - 1, disturbance_end + 168)

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    plot_timeline(ax1, sim_result.U_WT_series, "WT Availability Timeline", start_idx, end_idx)

    ax2 = plt.subplot(3, 1, 2)
    plot_timeline(ax2, sim_result.U_DG_series, "DG Availability Timeline", start_idx, end_idx)

    ax3 = plt.subplot(3, 1, 3)
    plot_timeline(ax3, sim_result.U_BAT_series, "Battery Availability Timeline", start_idx, end_idx)

    plt.tight_layout()
    plt.savefig("charts/DER_Availability_Timeline_WT_DG_BAT.png", dpi=200)
    plt.show()

    # PV 獨立一張
    plt.figure(figsize=(14, 8))
    plot_timeline(
        plt.gca(),
        sim_result.U_PV_series,
        "PV Availability Timeline (Isolated)",
        start_idx,
        end_idx,
    )
    plt.tight_layout()
    plt.savefig("charts/PV_Availability_Timeline.png", dpi=200)
    plt.show()

    print("完成圖表繪製!")
