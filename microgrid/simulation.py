# microgrid/simulation.py
import math
import random
from typing import Optional

import numpy as np

from .models import (
    MicrogridDesign,
    DisturbanceScenario,
    TimeSeriesInput,
    SimulationResult,
)


def simulate_microgrid_resilience(
    design: MicrogridDesign,
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    critical_load_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> SimulationResult:
    """
    新版微電網韌性模擬：
    - 主網停電：颱風期間(td~tfr) + 災後 MTTR 延長停電
    - 災前 24 小時自動把 SOC 充電到 90%
    - 災害期間：只供應 critical load
    - 主網可用時：使用 RE + (可選) BESS 放電 + Grid 補足
    - ★ 主電網恢復之後，柴油發電機強制關機（不再運轉）
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # ===== 基本時間序列 =====
    D_full = time_input.demand
    cf_WT_ts = time_input.cf_WT
    cf_PV_ts = time_input.cf_PV
    T_len = len(D_full)

    n_WT = len(design.P_WT)
    n_PV = len(design.P_PV)
    n_DG = len(design.P_DG)
    n_BAT = len(design.P_BAT)

    # ===== 初始可用狀態 =====
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

    # ===== 電池 SOC =====
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

    D_effective = [0.0] * T_len

    P_wt_all = [0.0] * T_len
    P_pv_all = [0.0] * T_len
    P_dg_all = [0.0] * T_len

    fuel_used = 0.0
    fuel_remaining = design.fuel_storage

    # ===== 災害相關時間 =====
    td = scenario.disturbance_start
    tfr = min(scenario.disturbance_end, T_len - 1)
    t_eval_end = min(T_len - 1, td + scenario.evaluation_horizon_hours)
    grid_back_time = tfr + scenario.grid_MTTR_hours  # ★ 災後延長停電的主網恢復時間

    # ===== PV/WT 災後 24 小時線性恢復 =====
    derate_PV = np.ones(T_len)
    derate_WT = np.ones(T_len)
    post_storm_window = 24

    for t in range(T_len):
        if tfr < t <= tfr + post_storm_window:
            alpha = (t - tfr) / post_storm_window
            derate_PV[t] = 0.6 + 0.4 * alpha
            derate_WT[t] = 0.6 + 0.4 * alpha

    # ===== 電池溫度/應力降額 =====
    temp_derate = np.ones(T_len)
    for t in range(T_len):
        h_wind = scenario.hazard.wind_speed[t]
        h_rain = scenario.hazard.rainfall[t]
        hazard_factor_tmp = min(1.0, (h_wind / 25.0) ** 2 + (h_rain / 80.0) ** 2)
        temp_derate[t] = 1.0 - 0.2 * hazard_factor_tmp

    # ======================================================
    #                     逐 小 時 模 擬
    # ======================================================
    for t in range(T_len):

        # ===== 判斷此時主網是否可用 =====
        in_storm = (td <= t <= tfr)
        pre_storm = (t < td) and ((td - t) <= 24)
        grid_available = (t < td) or (t > grid_back_time)

        Dt_full = D_full[t]

        # ===== 決定負載 & SOC 範圍 =====
        if grid_available:
            Dt = Dt_full
            soc_min_frac = design.B_min_soc_frac
            soc_max_frac = design.B_max_soc_frac
            if pre_storm:
                soc_max_frac = 0.90
        else:
            Dt = Dt_full * critical_load_ratio
            soc_min_frac = 0.10
            soc_max_frac = 0.90

        D_effective[t] = Dt

        # ★★★ 主電網恢復後（t > grid_back_time）強制 DG 關機，不再運轉 ★★★
        if t > grid_back_time:
            for i in range(n_DG):
                U_DG[i] = 0

        # ===== 設備故障（只有停電期間才會有 hazard-based 故障） =====
        if not grid_available:
            hazard_factor = min(
                1.0,
                (scenario.hazard.wind_speed[t] / 25.0) ** 2
                + (scenario.hazard.rainfall[t] / 80.0) ** 2,
            )
        else:
            hazard_factor = 0.0

        def update_failure(U, repair, base_p, MTTR):
            for i in range(len(U)):
                if U[i] == 1:
                    if random.random() < base_p * hazard_factor:
                        U[i] = 0
                        repair[i] = t + math.ceil(MTTR)
                else:
                    if repair[i] is not None and t >= repair[i]:
                        U[i] = 1

        update_failure(U_WT, repair_WT, scenario.base_p_damage_WT, scenario.MTTR_WT)
        update_failure(U_PV, repair_PV, scenario.base_p_damage_PV, scenario.MTTR_PV)
        # ★ DG 的故障/修復只在停電期間有意義，主網恢復後反正 U_DG 已被鎖 0
        if not grid_available:
            update_failure(U_DG, repair_DG, scenario.base_p_damage_DG, scenario.MTTR_DG)
        update_failure(U_BAT, repair_BAT, scenario.base_p_damage_BAT, scenario.MTTR_BAT)

        for i in range(n_WT):
            U_WT_series[i][t] = U_WT[i]
        for i in range(n_PV):
            U_PV_series[i][t] = U_PV[i]
        for i in range(n_DG):
            U_DG_series[i][t] = U_DG[i]
        for i in range(n_BAT):
            U_BAT_series[i][t] = U_BAT[i]

        # ===== 電池 SOC 傳遞 =====
        if t > 0:
            for i in range(n_BAT):
                B[i][t] = B[i][t - 1]

        for i in range(n_BAT):
            B_max_i = design.B_max[i]
            soc_min = soc_min_frac * B_max_i
            soc_max = soc_max_frac * B_max_i
            B[i][t] = min(max(B[i][t], soc_min), soc_max)

        # 災前 24 小時強制充電到 90%
        if pre_storm:
            for i in range(n_BAT):
                B[i][t] = max(B[i][t], 0.90 * design.B_max[i])

        # ===== WT / PV 出力 =====
        P_wt_t = sum(
            design.P_WT[i] * cf_WT_ts[t] * design.A_WT * derate_WT[t]
            for i in range(n_WT)
            if U_WT[i] == 1
        )

        P_pv_t = 0.0
        for i in range(n_PV):
            if U_PV[i] == 1:
                cf_eff = cf_PV_ts[t]
                if in_storm:
                    cf_eff = min(cf_eff, 0.05)
                P_pv_t += design.P_PV[i] * cf_eff * derate_PV[t]

        P_wt_all[t] = P_wt_t
        P_pv_all[t] = P_pv_t

        # ===== Diesel dispatch（只在停電期間使用，主網恢復後永遠不啟動） =====
        P_dg_t = 0.0
        if (not grid_available) and Dt > 0 and fuel_remaining > 0:
            residual = Dt - (P_wt_t + P_pv_t)
            for i in range(n_DG):
                if U_DG[i] == 0:
                    continue

                P_rated = design.P_DG[i]
                unit_min = P_rated * design.DG_min_loading
                unit_max = P_rated * design.DG_max_loading

                if residual <= 0:
                    break

                desired = min(residual, unit_max)
                if desired < unit_min:
                    if P_dg_t == 0.0:
                        desired = unit_min
                    else:
                        continue

                fuel_need = design.fuel_rate_max * (desired / P_rated)
                if fuel_remaining < fuel_need:
                    ratio = fuel_remaining / fuel_need
                    desired *= ratio
                    fuel_need = fuel_remaining

                P_dg_t += desired
                residual -= desired
                fuel_used += fuel_need
                fuel_remaining -= fuel_need

        P_dg_all[t] = P_dg_t

        # ===== A. 主網停電模式（Island Mode） =====
        if not grid_available:
            P_gen_no_batt = P_wt_t + P_pv_t + P_dg_t
            diff = Dt - P_gen_no_batt

            for i in range(n_BAT):
                P_charge[i][t] = 0.0
                P_discharge[i][t] = 0.0

            if diff < 0:
                # 充電
                surplus = -diff
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

                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_charge
                    P_cap_limit = (soc_max - B_prev) / max(design.eta_c, 1e-9)

                    P_max_charge = min(P_rating, P_c_rate, P_cap_limit)
                    charge = min(surplus, P_max_charge)
                    if charge <= 0:
                        continue

                    P_charge[i][t] = charge
                    B[i][t] = B_prev + charge * design.eta_c
                    surplus -= charge

                curtailment[t] = surplus
                Gt[t] = Dt
                Tt[t] = 0.0

            elif diff > 0:
                # 放電
                deficit = diff
                supply_b = 0.0

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

                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_discharge
                    max_ac = (B_prev - soc_min) * design.eta_d

                    P_max_discharge = min(P_rating, P_c_rate, max_ac)
                    discharge = min(deficit, P_max_discharge)
                    if discharge <= 0:
                        continue

                    P_discharge[i][t] = discharge
                    supply_b += discharge
                    deficit -= discharge
                    B[i][t] = max(
                        soc_min,
                        B_prev - discharge / max(design.eta_d, 1e-9),
                    )

                Gt[t] = P_gen_no_batt + supply_b
                Tt[t] = max(0.0, Dt - Gt[t])

            else:
                Gt[t] = Dt
                Tt[t] = 0.0

            service_level[t] = min(1.0, Gt[t] / Dt) if Dt > 0 else 1.0

        # ===== B. 主網可用模式（Grid-connected Mode） =====
        else:
            P_re = P_wt_t + P_pv_t
            served_by_RE = min(P_re, Dt)
            residual = Dt - served_by_RE
            surplus_re = max(0.0, P_re - served_by_RE)

            for i in range(n_BAT):
                P_charge[i][t] = 0.0
                P_discharge[i][t] = 0.0

            supply_b = 0.0
            if residual > 0:
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if residual <= 0:
                        break

                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_min = soc_min_frac * B_max_i

                    if B_prev <= soc_min:
                        continue

                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_discharge
                    max_ac = (B_prev - soc_min) * design.eta_d

                    P_max_discharge = min(P_rating, P_c_rate, max_ac)
                    discharge = min(residual, P_max_discharge)
                    if discharge <= 0:
                        continue

                    P_discharge[i][t] = discharge
                    supply_b += discharge
                    residual -= discharge
                    B[i][t] = max(
                        soc_min,
                        B_prev - discharge / max(design.eta_d, 1e-9),
                    )

            # 主網補足剩餘負載
            grid_supply = max(0.0, residual)

            # surplus RE → 充電
            if surplus_re > 0:
                for i in range(n_BAT):
                    if U_BAT[i] == 0:
                        continue
                    if surplus_re <= 0:
                        break

                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    soc_max = soc_max_frac * B_max_i

                    if B_prev >= soc_max:
                        continue

                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_charge
                    P_cap_limit = (soc_max - B_prev) / max(design.eta_c, 1e-9)

                    P_max_charge = min(P_rating, P_c_rate, P_cap_limit)
                    charge = min(surplus_re, P_max_charge)
                    if charge <= 0:
                        continue

                    P_charge[i][t] = charge
                    B[i][t] = B_prev + charge * design.eta_c
                    surplus_re -= charge

            curtailment[t] = surplus_re
            Gt[t] = Dt
            Tt[t] = 0.0
            service_level[t] = 1.0

        Pt[t] = P_wt_t + P_pv_t + P_dg_t + sum(design.P_BAT)

    # ======================================================
    #          韌性指標：Invulnerability + Resilience Curve
    # ======================================================
    P_td = Pt[td] if 0 <= td < T_len else 0.0
    P_ts = Pt[tfr] if 0 <= tfr < T_len else 0.0
    invulnerability = (P_ts / P_td) if P_td > 0 else 0.0

    num = sum(1.0 - service_level[t] for t in range(td, t_eval_end + 1))
    den = max(1, t_eval_end - td + 1)
    resilience_curve = max(0.0, 1.0 - num / den)

    recovery_time_h = None
    for t in range(tfr, t_eval_end + 1):
        if service_level[t] >= 0.99:
            if all(service_level[k] >= 0.95 for k in range(t, t_eval_end + 1)):
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
