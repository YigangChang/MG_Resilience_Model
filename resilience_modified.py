"""
Modified microgrid resilience simulation incorporating
pre‑disaster operation, storm‑time behaviour, critical load
considerations and enhanced hazard failure models.

This script adapts the original simulation to:
  * Maintain battery state of charge (SOC) between 50% and 80% by
    adjusting the design parameters. Batteries are initialised at
    80% SOC so they are ready for use when the storm begins.
  * Represent critical load as 20% of the total demand. Service
    levels are therefore computed relative to this critical load
    rather than the full demand, meaning a full service level
    corresponds to meeting 20% of the demand during the disaster.
  * Increase diesel generator capacity by 15 MW. The design adds
    three additional 5 MW diesel units to the existing fleet.
  * Increase the diesel fuel storage to cover roughly seven days
    of operation at rated load across all diesel generators.
  * Limit photovoltaic output to a maximum of 5% of rated
    capacity during the storm window. Wind turbines and PV
    continue to follow the given capacity factors outside the
    storm.
  * Adjust the baseline failure probabilities for wind and PV
    equipment to lie within the specified ranges (0.05–0.15 for
    wind and 0.03–0.10 for PV) and enhance the hazard factor
    calculation to increase sensitivity to high wind speeds and
    rainfall.

The rest of the architecture—cost computation, resiliency
metrics, plotting—remains largely unchanged.
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
# 2. Part 2: 資料結構定義
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

    B_min_soc_frac: float
    B_max_soc_frac: float
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
    demand: List[float]

    B: List[List[float]]
    P_charge: List[List[float]]
    P_discharge: List[List[float]]
    curtailment: List[float]

    fuel_used: float
    service_level: List[float]

    # 新增輸出：風力、太陽能與柴油發電機每小時輸出功率序列，以及
    # 電池總充電與總放電功率（各電池單位求和）。這些可用於後續
    # 視覺化或分析。
    P_wt: List[float]
    P_pv: List[float]
    P_dg: List[float]


# =========================================================
# 3. Part 2: 核心韌性模擬 simulate_microgrid_resilience
# =========================================================

def simulate_microgrid_resilience(
    design: MicrogridDesign,
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    critical_load_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> SimulationResult:
    """
    新版韌性模擬：
    - 使用 5~11 月長期逐時資料
    - 故障依 hazard (風速+雨量) 逐時機率判定
    - 柴油機需求追隨 + min/max loading
    - PV/WT 災後降額（24 小時內線性恢復）
    - 電池 DoD / C-rate / 溫度降額
    - 韌性指標：Invulnerability + IEEE-style Resiliency Curve

    此修改版本加入了：
    * 在 storm 期間 PV 的容量因子限制為不超過 5%。
    * 服務水準參考 critical_load_ratio，用以表示僅需供應部分重要負載。
    * hazard factor 計算更敏感於高風速與強降雨。
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Demand and capacity factors from the time series input
    D_full = time_input.demand  # full demand
    cf_WT_ts = time_input.cf_WT
    cf_PV_ts = time_input.cf_PV
    T_len = len(D_full)

    # Number of devices by type
    n_WT = len(design.P_WT)
    n_PV = len(design.P_PV)
    n_DG = len(design.P_DG)
    n_BAT = len(design.P_BAT)

    # Initialise device availability and repair timers
    U_WT = [1] * n_WT
    U_PV = [1] * n_PV
    U_DG = [1] * n_DG
    U_BAT = [1] * n_BAT

    repair_WT = [None] * n_WT
    repair_PV = [None] * n_PV
    repair_DG = [None] * n_DG
    repair_BAT = [None] * n_BAT

    # Battery state of charge (SOC) matrix
    B = [[0.0] * T_len for _ in range(n_BAT)]
    for i in range(n_BAT):
        # initial SOC is provided in design.B_init (already set to 80% of B_max)
        B[i][0] = design.B_init[i]

    P_charge = [[0.0] * T_len for _ in range(n_BAT)]
    P_discharge = [[0.0] * T_len for _ in range(n_BAT)]
    curtailment = [0.0] * T_len

    Pt = [0.0] * T_len
    Gt = [0.0] * T_len
    Tt = [0.0] * T_len
    service_level = [1.0] * T_len

    # Effective demand array (scaled during storm)
    D_effective = [0.0] * T_len

    # Track hourly generation from wind turbines, PV and diesel generators
    P_wt_all = [0.0] * T_len
    P_pv_all = [0.0] * T_len
    P_dg_all = [0.0] * T_len

    fuel_used = 0.0
    fuel_remaining = design.fuel_storage

    # Time indices for storm start and end
    td = scenario.disturbance_start
    tfr = min(scenario.disturbance_end, T_len - 1)
    t_eval_end = min(T_len - 1, td + scenario.evaluation_horizon_hours)

    # Temperature/stress derating factor (affects battery C-rate and generator derating)
    temp_derate = np.ones(T_len, dtype=float)
    for t in range(T_len):
        h_wind = scenario.hazard.wind_speed[t]
        h_rain = scenario.hazard.rainfall[t]
        # Hazard factor used for derating battery output based on wind and rain
        hazard_factor = min(1.0, (h_wind / 30.0) + (h_rain / 80.0))
        temp_derate[t] = 1.0 - 0.2 * hazard_factor  # at most derate by 20%

    # PV/WT post-storm derate: linear recovery over 24h after storm ends
    derate_PV = np.ones(T_len, dtype=float)
    derate_WT = np.ones(T_len, dtype=float)
    post_storm_window = 24
    for t in range(T_len):
        if tfr < t <= tfr + post_storm_window:
            alpha = (t - tfr) / post_storm_window
            factor = 0.6 + 0.4 * alpha  # from 60% to 100%
            derate_PV[t] = factor
            derate_WT[t] = factor

    # Run the hourly simulation
    for t in range(T_len):
        # Determine the effective demand. Outside of the storm window the
        # microgrid attempts to serve the full demand. During the
        # disturbance period only the critical load (a fraction of the
        # demand) must be served. This reflects load shedding down to
        # essential services.
        Dt_full = D_full[t]
        if td <= t <= tfr:
            Dt = Dt_full * critical_load_ratio
        else:
            Dt = Dt_full

        # record the effective demand for output and visualisation
        D_effective[t] = Dt

        # 1) Update failure/repair states based on hazard during the storm
        if td <= t <= tfr:
            h_wind = scenario.hazard.wind_speed[t]
            h_rain = scenario.hazard.rainfall[t]
            # Adjusted hazard factor emphasising high wind/rain using squared terms
            hazard_factor = min(1.0, (h_wind / 30.0) ** 2 + (h_rain / 80.0) ** 2)
        else:
            hazard_factor = 0.0

        # Update WT availability
        for i in range(n_WT):
            if U_WT[i] == 1:
                p_fail = scenario.base_p_damage_WT * hazard_factor
                if random.random() < p_fail:
                    U_WT[i] = 0
                    repair_WT[i] = t + math.ceil(scenario.MTTR_WT)
            else:
                if repair_WT[i] is not None and t >= repair_WT[i]:
                    U_WT[i] = 1

        # Update PV availability
        for i in range(n_PV):
            if U_PV[i] == 1:
                p_fail = scenario.base_p_damage_PV * hazard_factor
                if random.random() < p_fail:
                    U_PV[i] = 0
                    repair_PV[i] = t + math.ceil(scenario.MTTR_PV)
            else:
                if repair_PV[i] is not None and t >= repair_PV[i]:
                    U_PV[i] = 1

        # Update DG availability
        for i in range(n_DG):
            if U_DG[i] == 1:
                p_fail = scenario.base_p_damage_DG * hazard_factor
                if random.random() < p_fail:
                    U_DG[i] = 0
                    repair_DG[i] = t + math.ceil(scenario.MTTR_DG)
            else:
                if repair_DG[i] is not None and t >= repair_DG[i]:
                    U_DG[i] = 1

        # Update BAT availability
        for i in range(n_BAT):
            if U_BAT[i] == 1:
                p_fail = scenario.base_p_damage_BAT * hazard_factor
                if random.random() < p_fail:
                    U_BAT[i] = 0
                    repair_BAT[i] = t + math.ceil(scenario.MTTR_BAT)
            else:
                if repair_BAT[i] is not None and t >= repair_BAT[i]:
                    U_BAT[i] = 1

        # 2) Propagate SOC from previous hour
        if t > 0:
            for i in range(n_BAT):
                B[i][t] = B[i][t - 1]

        # Enforce SOC bounds according to design
        for i in range(n_BAT):
            B_max_i = design.B_max[i]
            soc_min = design.B_min_soc_frac * B_max_i
            soc_max = design.B_max_soc_frac * B_max_i
            B[i][t] = min(max(B[i][t], soc_min), soc_max)

        # --------------------------------------------------
        # Pre‑storm pre‑charge strategy: ensure batteries
        # are fully charged (to their maximum allowed SOC) in
        # the 24 hours leading up to the storm. If the current
        # time step is within 24 hours before the storm starts,
        # force the battery SOC to the upper limit. This
        # represents a preemptive charging action so that the
        # storage is ready when the storm begins.
        # --------------------------------------------------
        if td is not None and t < td and (td - t) <= 24:
            for i in range(n_BAT):
                B[i][t] = design.B_max[i] * design.B_max_soc_frac

        # 3) Compute wind and solar output
        P_wt_t = 0.0
        for i in range(n_WT):
            if U_WT[i] == 1:
                P_wt_t += design.P_WT[i] * cf_WT_ts[t] * design.A_WT * derate_WT[t]

        P_pv_t = 0.0
        for i in range(n_PV):
            if U_PV[i] == 1:
                # During the storm PV output is limited to 5% of rated capacity
                cf_pv_eff = cf_PV_ts[t]
                if td <= t <= tfr:
                    cf_pv_eff = min(cf_pv_eff, 0.05)
                P_pv_t += design.P_PV[i] * cf_pv_eff * derate_PV[t]

        # record generation from wind and PV for this hour
        P_wt_all[t] = P_wt_t
        P_pv_all[t] = P_pv_t

        # 4) Diesel generator dispatch (follows demand, min/max loading)
        P_dg_t = 0.0
        dg_min = design.DG_min_loading
        dg_max = design.DG_max_loading

        residual_after_RE = max(0.0, Dt - (P_wt_t + P_pv_t))

        if fuel_remaining > 0 and residual_after_RE > 0:
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
                    # Must operate at minimum loading if unit is on
                    if P_dg_t == 0.0:
                        actual = unit_min
                    else:
                        continue
                else:
                    actual = desired

                # Fuel required at full load: fuel_rate_max * (actual / P_rated)
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

        # record diesel generation for this hour
        P_dg_all[t] = P_dg_t

        # 5) Battery charge/discharge logic with derating
        temp_factor = temp_derate[t]
        P_gen_without_batt = P_wt_t + P_pv_t + P_dg_t
        T_without_batt = Dt - P_gen_without_batt  # positive = deficit, negative = surplus

        # Reset battery dispatch this hour
        for i in range(n_BAT):
            P_charge[i][t] = 0.0
            P_discharge[i][t] = 0.0

        # Surplus: attempt to charge batteries
        if T_without_batt < 0:
            surplus = -T_without_batt
            for i in range(n_BAT):
                if U_BAT[i] == 0:
                    continue
                if surplus <= 0:
                    break

                B_prev = B[i][t]
                B_max_i = design.B_max[i]
                soc_max = design.B_max_soc_frac * B_max_i
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
            # supply fully meets effective demand
            Gt[t] = Dt
            Tt[t] = 0.0

        # Deficit: discharge batteries
        elif T_without_batt > 0:
            deficit = T_without_batt
            supplied_by_batt = 0.0

            for i in range(n_BAT):
                if U_BAT[i] == 0:
                    continue
                if deficit <= 0:
                    break

                B_prev = B[i][t]
                B_max_i = design.B_max[i]
                soc_min = design.B_min_soc_frac * B_max_i
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

        # Balanced: no battery action
        else:
            # balanced generation exactly meets the effective demand
            Gt[t] = Dt
            Tt[t] = 0.0
            curtailment[t] = 0.0

        Pt[t] = P_wt_t + P_pv_t + P_dg_t + sum(design.P_BAT)

        # Compute service level. We use the effective demand as the
        # denominator: outside the storm this is the full demand; during
        # the storm it is the critical load. Clamp to 1.0 when supply
        # exceeds demand. If demand is zero, the service level is 1 by
        # definition.
        if Dt > 0:
            service_level[t] = min(1.0, Gt[t] / Dt)
        else:
            service_level[t] = 1.0

    # ====================================================
    # 6. 韌性指標：Invulnerability + Resiliency Curve
    # ====================================================
    # 衡量災害前後微電網可運轉能力的比值
    P_td = Pt[td] if 0 <= td < T_len else 0.0
    P_ts = Pt[tfr] if 0 <= tfr < T_len else 0.0
    invulnerability = (P_ts / P_td) if P_td > 0 else 0.0

    # 災後一段時間內平均服務水準的比例（依 IEEE/NREL 定義）
    num = 0.0
    den = 0.0
    for t in range(td, t_eval_end + 1):
        num += (1.0 - service_level[t])
        den += 1.0
    resilience_curve = 1.0 - num / den if den > 0 else 0.0

    # 從災害發生到系統恢復到穩定 0.99 水準的時間
    recovery_time_h = None
    for t in range(tfr, t_eval_end + 1):
        if service_level[t] >= 0.99:
            # check that service_level stays above 0.95 until end of evaluation
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
    )


# =========================================================
# 4. 成本模組：LCOE / LCOED
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


def _pad_to_horizon(lst: List[float], p: int) -> List[float]:
    if len(lst) >= p:
        return lst[:p]
    if not lst:
        return [0.0] * p
    return lst + [lst[-1]] * (p - len(lst))


def compute_LCOE(
    cost: CostParameters,
    Ey_year: List[float],
    fuel_used_sim: float,
    days_simulated: float = 214.0,
    fuel_price_per_gal: float = 97.0,
) -> float:
    """
    Compute the Levelised Cost of Energy (LCOE).

    Diesel fuel cost is computed from fuel_used_sim and fuel_price_per_gal
    scaled to one year. Capital costs, salvage value and O&M are summed
    across all device types.
    """
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

    def pad(lst):
        if len(lst) >= p:
            return lst[:p]
        if not lst:
            return [0.0] * p
        return lst + [lst[-1]] * (p - len(lst))

    M_WT_y = pad(cost.M_WT)
    M_PV_y = pad(cost.M_PV)
    M_DG_y = pad(cost.M_DG)
    M_BAT_y = pad(cost.M_BAT)

    # Fuel cost scaled to a full year
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
    """
    Compute the Levelised Cost of Delivered Energy (LCOED).

    Denominator uses annualised demand (Dy_year). Otherwise identical to LCOE.
    """
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

    def pad(lst):
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


# =========================================================
# 5. 多方案評估 evaluate_designs（可之後用）
# =========================================================

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

        LCOE = compute_LCOE(
            cost=base_cost,
            Ey_year=Ey_year,
            fuel_used_sim=sim.fuel_used,
            days_simulated=days_in_dataset,
            fuel_price_per_gal=97.0,
        )

        LCOED = compute_LCOED(
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
                "LCOE": LCOE,
                "LCOED": LCOED,
                "fuel_used": sim.fuel_used,
            }
        )

    return results


# =========================================================
# 6. 主程式
# =========================================================

if __name__ == "__main__":
    # Read input time series data
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

    # Find 8/7 00:00 as the hurricane start time
    disturbance_start = None
    for idx in range(total_hours):
        if months_series[idx] == 8 and days_series[idx] == 7 and hours_of_day[idx] == 0:
            disturbance_start = idx
            break
    if disturbance_start is None:
        raise RuntimeError("Cannot find 8/7 00:00 disturbance start; please check time series data.")

    disturbance_duration_h = 72  # 3 days storm duration
    disturbance_end = min(total_hours - 1, disturbance_start + disturbance_duration_h)

    # Set scenario with enhanced damage probabilities (within specified ranges)
    scenario_hurricane = DisturbanceScenario(
        name="Morakot_like",
        disturbance_start=disturbance_start,
        disturbance_end=disturbance_end,
        base_p_damage_WT=0.10,  # midpoint of 0.05–0.15
        base_p_damage_PV=0.065,  # midpoint of 0.03–0.10
        base_p_damage_DG=0.01,
        base_p_damage_BAT=0.01,
        MTTR_WT=72.0,
        MTTR_PV=31.0,
        MTTR_DG=289.0,
        MTTR_BAT=3.0,
        hazard=hazard,
        evaluation_horizon_hours=168,
    )

    # Example microgrid design
    # Increase DG capacity: original had 3x5 MW; add three more 5 MW units
    P_DG_units = [5000.0] * 6  # 6 units of 5 MW each = 30 MW total

    # Battery design: maintain 50–80% SOC (B_min_soc_frac=0.5, B_max_soc_frac=0.8)
    B_max_list = [4000.0] * 10
    B_init_list = [b * 0.8 for b in B_max_list]  # initialise at 80% of B_max

    # Estimate fuel storage for 7 days of operation at full load for all DG units
    fuel_rate_max = 350.0
    days_fuel = 7
    fuel_storage_required = fuel_rate_max * 24 * days_fuel * len(P_DG_units)

    design_example = MicrogridDesign(
        P_WT=[2000.0] * 5,
        P_PV=[1000.0] * 20,
        P_DG=P_DG_units,
        P_BAT=[2000.0] * 10,
        B_max=B_max_list,
        B_init=B_init_list,
        eta_c=0.97,
        eta_d=0.98,
        C_WT=0.22,
        C_PV=0.36,
        A_WT=0.98,
        fuel_rate_max=fuel_rate_max,
        fuel_storage=fuel_storage_required,
        DG_min_loading=0.3,
        DG_max_loading=0.8,
        B_min_soc_frac=0.5,
        B_max_soc_frac=0.8,
        C_rate_charge=0.5,
        C_rate_discharge=1.0,
    )

    # Example cost parameters (unchanged from original)
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

    # Run the simulation for the example design
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
    # 7. Visualization: Hourly profiles & Resilience curve
    # ====================================================
    # Determine indices for plotting: 24h before storm to evaluation horizon after storm
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

    # Extract generation and battery charge/discharge for the selected window
    P_wt_window = np.array(sim_result.P_wt)[idx_start:idx_end + 1]
    P_pv_window = np.array(sim_result.P_pv)[idx_start:idx_end + 1]
    P_dg_window = np.array(sim_result.P_dg)[idx_start:idx_end + 1]
    # Sum battery charge and discharge across all battery units
    P_charge_total_window = np.sum(np.array(sim_result.P_charge)[:, idx_start:idx_end + 1], axis=0)
    P_discharge_total_window = np.sum(np.array(sim_result.P_discharge)[:, idx_start:idx_end + 1], axis=0)

    # Plot 1: Demand vs Served vs Unserved load
    plt.figure(figsize=(12, 5))
    plt.plot(timestamp_window, demand_window, label="Demand [kW]")
    plt.plot(timestamp_window, Gt_window, label="Served Load Gt [kW]")
    plt.plot(timestamp_window, Tt_window, label="Unserved Load Tt [kW]")
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--", label="Disturbance start")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--", label="Disturbance end")
    plt.xlabel("Time")
    plt.ylabel("Power / kW")
    plt.title("Hourly Demand, Served Load, and Unserved Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Battery SOC average
    plt.figure(figsize=(12, 4))
    plt.plot(timestamp_window, B_avg_window)
    plt.axvline(df["timestamp"].iloc[disturbance_start], linestyle="--")
    plt.axvline(df["timestamp"].iloc[disturbance_end], linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Battery SOC (average across batteries) [kWh]")
    plt.title("Hourly Battery SOC (Average)")
    plt.tight_layout()
    plt.show()

    # Plot 3: Resilience curve (service level over time)
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
    plt.show()

    # ---------- 圖 4：各機組逐時發電量與電池充放電 ----------
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
    plt.show()
    print("完成圖表繪製!")