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
