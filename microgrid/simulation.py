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

    - 主網停電：顯式停電區間為 [td, tfr]，並延長至 tfr + grid_MTTR_hours
    - 災前 24 小時：主網可用時，電池 SOC 強制維持 90%（視為主網幫忙充電）
        * 此期間不執行每日放電 / 一般 EMS 控制
    - 災害 + 停電期間（主網不可用）：
        * 只供應 critical load = critical_load_ratio * demand
        * 供電順序：WT/PV -> DG -> BESS -> Unserved
        (*供電順序：WT/PV -> BESS -> DG -> Unserved??) 可嘗試?
        * 電池 SOC 範圍：10% ~ 90%
    - 主網可用期間（含非災期與復電後）：
        * 負載 = 全部 demand
        * 平常 SOC 範圍：B_min_soc_frac ~ B_max_soc_frac（例如 20%~80%）
        * 每日 EMS：
            - 00:00–06:00：強制充到 80% SOC（先用 RE，多的用 Grid）
            - 18:00–22:00：放電到 20% SOC（用 BESS 供應部分負載）
            - 其它時間：維持「RE 優先，其次 BESS，最後 Grid」的邏輯
        * 主網可用時，使用者無缺電：service_level = 1
    - 主網恢復後（t > grid_back_time）：柴油機強制不再啟動（U_DG = 0）
    """

    # ===== 隨機種子 =====
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # ===== 基本時間序列 =====
    D_full = time_input.demand
    cf_WT_ts = time_input.cf_WT
    cf_PV_ts = time_input.cf_PV
    hours_ts = time_input.hours  # 0~23
    T_len = len(D_full)

    n_WT = len(design.P_WT)
    n_PV = len(design.P_PV)
    n_DG = len(design.P_DG)
    n_BAT = len(design.P_BAT)

    # ===== 初始設備狀態 =====
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

    # ===== 電池 SOC 初始 =====
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

    # ===== 災害時間 & 評估視窗 =====
    td = scenario.disturbance_start
    tfr = min(scenario.disturbance_end, T_len - 1)
    t_eval_end = min(T_len - 1, td + scenario.evaluation_horizon_hours)
    grid_back_time = tfr + scenario.grid_MTTR_hours  # 延長停電後主網恢復時間

    # ===== PV/WT 災後 24 小時線性恢復 =====
    derate_PV = np.ones(T_len)
    derate_WT = np.ones(T_len)
    post_storm_window = 24
    for t in range(T_len):
        if tfr < t <= tfr + post_storm_window:
            alpha = (t - tfr) / post_storm_window
            derate_PV[t] = 0.6 + 0.4 * alpha
            derate_WT[t] = 0.6 + 0.4 * alpha

    # ===== 電池溫度/應力降額（用於 C-rate） =====
    temp_derate = np.ones(T_len)
    for t in range(T_len):
        h_wind = scenario.hazard.wind_speed[t]
        h_rain = scenario.hazard.rainfall[t]
        hazard_factor_tmp = min(
            1.0,
            (h_wind / 25.0) ** 2 + (h_rain / 80.0) ** 2,
        )
        temp_derate[t] = 1.0 - 0.2 * hazard_factor_tmp

    # ======================================================
    #                     逐 小 時 模 擬
    # ======================================================
    for t in range(T_len):
        # --- 狀態判斷 ---
        in_storm = (td <= t <= tfr)
        pre_storm = (t < td) and ((td - t) <= 24)
        grid_available = (t < td) or (t > grid_back_time)

        Dt_full = D_full[t]
        hour = hours_ts[t] if hours_ts is not None else (t % 24)

        # --- 決定負載 & SOC 範圍 ---
        if grid_available:
            # 主網可用：全負載
            Dt = Dt_full
            soc_min_frac = design.B_min_soc_frac   # eg. 0.2
            soc_max_frac = design.B_max_soc_frac   # eg. 0.8
            if pre_storm:
                # 災前 24h：SOC 上限允許到 90%
                soc_max_frac = 0.90
        else:
            # 主網停電：只供應 critical load
            Dt = Dt_full * critical_load_ratio
            soc_min_frac = 0.10
            soc_max_frac = 0.90

        D_effective[t] = Dt

        # --- 主網恢復後強制 DG 不再運轉 ---
        if t > grid_back_time:
            for i in range(n_DG):
                U_DG[i] = 0

        # --- 設備故障（只在停電期間有 hazard-based 故障）---
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
        if not grid_available:
            update_failure(U_DG, repair_DG, scenario.base_p_damage_DG, scenario.MTTR_DG)
        update_failure(U_BAT, repair_BAT, scenario.base_p_damage_BAT, scenario.MTTR_BAT)

        # 紀錄可用狀態
        for i in range(n_WT):
            U_WT_series[i][t] = U_WT[i]
        for i in range(n_PV):
            U_PV_series[i][t] = U_PV[i]
        for i in range(n_DG):
            U_DG_series[i][t] = U_DG[i]
        for i in range(n_BAT):
            U_BAT_series[i][t] = U_BAT[i]

        # --- 電池 SOC 傳遞 ---
        if t > 0:
            for i in range(n_BAT):
                B[i][t] = B[i][t - 1]

        # SOC 邊界
        for i in range(n_BAT):
            B_max_i = design.B_max[i]
            soc_min = soc_min_frac * B_max_i
            soc_max = soc_max_frac * B_max_i
            B[i][t] = min(max(B[i][t], soc_min), soc_max)

        # --- WT / PV 出力 ---
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

        # 預設 DG 為 0（以防早期 return/continue）
        P_dg_t = 0.0

        # === 災前 24 小時：強制電池維持 90% SOC，略過 EMS / DG / Island 邏輯 ===
        if pre_storm and grid_available:
            target_soc = 0.90
            for i in range(n_BAT):
                B_max_i = design.B_max[i]
                target_energy = target_soc * B_max_i

                # 主網補足：若 SOC < 90%，直接拉到 90%
                if B[i][t] < target_energy:
                    B[i][t] = target_energy

                # 當小時不允許充放電（避免 EMS override）
                P_charge[i][t] = 0.0
                P_discharge[i][t] = 0.0

            # 使用者角度：Grid + RE 足以滿足 Dt
            Gt[t] = Dt
            Tt[t] = 0.0
            curtailment[t] = 0.0
            service_level[t] = 1.0

            P_dg_all[t] = 0.0
            Pt[t] = P_wt_t + P_pv_t + 0.0 + sum(design.P_BAT)
            continue  # 跳過後續邏輯，進入下一個小時

        # --- Diesel dispatch（只在停電期間使用）---
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
            diff = Dt - P_gen_no_batt  # >0 代表缺電，<0 代表多餘

            # reset 當小時電池功率
            for i in range(n_BAT):
                P_charge[i][t] = 0.0
                P_discharge[i][t] = 0.0

            if diff < 0:
                # --- 多餘能量：充電 ---
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
                # --- 缺電：放電 ---
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
                # 剛好平衡
                Gt[t] = Dt
                Tt[t] = 0.0

            service_level[t] = min(1.0, Gt[t] / Dt) if Dt > 0 else 1.0

        # ===== B. 主網可用模式（Grid-connected Mode） =====
        else:
            # reset 電池功率
            for i in range(n_BAT):
                P_charge[i][t] = 0.0
                P_discharge[i][t] = 0.0

            # ---------------------------
            # DAILY CHARGE / DISCHARGE EMS
            # ---------------------------
            CHARGE_START = 0      # 00:00
            CHARGE_END   = 6      # [0,6)
            DISCHARGE_START = 18  # 18:00
            DISCHARGE_END   = 22  # [18,22)

            target_soc_charge = 0.80
            target_soc_discharge = 0.20

            P_re = P_wt_t + P_pv_t

            # -------- CASE 1：充電時段（每天充到 80%） --------
            if CHARGE_START <= hour < CHARGE_END:
                # 先由 RE 供負載
                served_by_RE = min(P_re, Dt)
                residual_load = Dt - served_by_RE
                # RE 剩餘可用於充電
                surplus = max(0.0, P_re - served_by_RE)

                for i in range(n_BAT):
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    target_energy = target_soc_charge * B_max_i

                    if B_prev >= target_energy:
                        continue

                    cap_limit = (target_energy - B_prev) / max(design.eta_c, 1e-9)
                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_charge
                    P_max = min(P_rating, P_c_rate, cap_limit)

                    # 先用 RE surplus 充
                    charge_from_RE = min(surplus, P_max)
                    B[i][t] = B_prev + charge_from_RE * design.eta_c
                    P_charge[i][t] += charge_from_RE
                    surplus -= charge_from_RE

                    # 不足的部分用 Grid 補
                    if B[i][t] < target_energy:
                        remaining_energy = target_energy - B[i][t]
                        charge_from_grid = min(
                            P_max - charge_from_RE,
                            remaining_energy / max(design.eta_c, 1e-9),
                        )
                        B[i][t] += charge_from_grid * design.eta_c
                        P_charge[i][t] += charge_from_grid

                # 使用者角度：Grid + RE 足以滿足 Dt
                Gt[t] = Dt
                Tt[t] = 0.0
                curtailment[t] = max(0.0, surplus)
                service_level[t] = 1.0

            # -------- CASE 2：放電時段（每天放到 20%） --------
            # ⚠ 災害前 24 小時不執行放電（已在 pre_storm 分支中被 continue 掉）
            elif (DISCHARGE_START <= hour < DISCHARGE_END) and grid_available:
                served_by_RE = min(P_re, Dt)
                residual = Dt - served_by_RE
                supply_b = 0.0

                for i in range(n_BAT):
                    B_prev = B[i][t]
                    B_max_i = design.B_max[i]
                    min_energy = target_soc_discharge * B_max_i

                    if B_prev <= min_energy or residual <= 0:
                        continue

                    P_rating = design.P_BAT[i] * temp_derate[t]
                    P_c_rate = B_max_i * design.C_rate_discharge
                    energy_limit = (B_prev - min_energy) * design.eta_d

                    P_max = min(P_rating, P_c_rate, energy_limit)
                    discharge = min(residual, P_max)
                    if discharge <= 0:
                        continue

                    P_discharge[i][t] = discharge
                    B[i][t] = B_prev - discharge / max(design.eta_d, 1e-9)
                    residual -= discharge
                    supply_b += discharge

                # 主網補足剩餘
                grid_supply = max(0.0, residual)

                Gt[t] = Dt
                Tt[t] = 0.0
                curtailment[t] = 0.0
                service_level[t] = 1.0

            # -------- CASE 3：一般 Grid-connected 狀態 --------
            else:
                P_re = P_wt_t + P_pv_t
                served_by_RE = min(P_re, Dt)
                residual = Dt - served_by_RE
                surplus_re = max(0.0, P_re - served_by_RE)

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

                grid_supply = max(0.0, residual)

                # surplus RE 充電
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
                        cap_limit = (soc_max - B_prev) / max(design.eta_c, 1e-9)

                        P_max = min(P_rating, P_c_rate, cap_limit)
                        charge = min(surplus_re, P_max)
                        if charge <= 0:
                            continue

                        P_charge[i][t] = charge
                        B[i][t] = B_prev + charge * design.eta_c
                        surplus_re -= charge

                curtailment[t] = surplus_re
                Gt[t] = Dt
                Tt[t] = 0.0
                service_level[t] = 1.0

        # ===== 統一計算 Pt（每一小時結尾） =====
        Pt[t] = P_wt_t + P_pv_t + P_dg_t + sum(design.P_BAT)

    # ======================================================
    #                       韌性指標
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
    
    # 1. EENS (Energy Not Served)
    EENS = sum(Tt[t] for t in range(td, t_eval_end + 1))
    EENS = round(EENS, 1)
    # 2. LOLE (Loss of Load Expectation = 缺電小時數)
    LOLE = sum(1 for t in range(td, t_eval_end + 1) if Tt[t] > 1e-6)

    # 3. Critical Load Survival Time
    survival = 0
    for t in range(td, t_eval_end + 1):
        if Tt[t] <= 1e-6:   # 完全供應 critical load
            survival += 1
        else:
            break
    critical_load_survival_time = survival
    
    # 4. Fuel Sustainability = 若燃料耗盡率維持災期平均，其可撐多久
    fuel_sustainability_h = 9999   # default: 幾乎無限制
    if fuel_used > 0:
        storm_hours = max(1, tfr - td + 1)
        avg_hourly_fuel = fuel_used / storm_hours
        if avg_hourly_fuel > 0:
            fuel_sustainability_h = fuel_remaining / avg_hourly_fuel
            fuel_sustainability_h = round(fuel_sustainability_h, 1)
    
    # 5. EENS Ratio
    # 災害期間總 critical load 能量（分母）
    total_critical_energy = sum(
    D_effective[t] for t in range(td, t_eval_end + 1))
    if total_critical_energy > 0:
        EENS_ratio = EENS / total_critical_energy
        EENS_ratio = round(EENS_ratio, 2)
    else:
        EENS_ratio = 0.0

    

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
        
        EENS=EENS,
        LOLE=LOLE,
        critical_load_survival_time=critical_load_survival_time,
        fuel_sustainability_h=fuel_sustainability_h,
        EENS_ratio=EENS_ratio,
    )
