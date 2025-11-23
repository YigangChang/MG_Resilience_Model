"""
依需求，你可以：
1. 定義微電網配置 (DER 容量、成本、MTTR、損壞機率…)
2. 定義負載與容量因子時間序列
3. 呼叫 simulate_microgrid_resilience() 取得：
   - Invulnerability, Recovery, Resilience
   - 各時間步的發電、未滿足負載、電池 SOC
4. 呼叫 compute_LCOE(), compute_LCOED() 計算成本指標
5. 呼叫 evaluate_designs() 對多組配置做簡單設計搜尋
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random
import pandas as pd

# ==============================
# 使用者提供的電池充電函式（原樣保留）
# ==============================

def calculate_charge_power_physical_v1(B_prev, B_max, T, P_bat, eta_c, q, td, tfr):
    """
    A: 真實電網情境：過剩功率無法跨時間 t→t+1

    B_prev: 2D list, shape (q, time_len)，前一時間點 SOC (kWh)
    B_max : 1D list, 最大容量 (kWh)
    T     : 1D list, 負值表示過剩功率 (kW)
    P_bat : 1D list, 額定充電功率 (kW)
    eta_c : 充電效率 (0~1)
    q     : 電池數量
    td, tfr: 起訖時間 index（含）
    """

    time_len = len(T)

    # 充電功率 (kW)
    P_charge = [[0.0 for _ in range(time_len)] for _ in range(q)]

    # 棄電 (kW)
    curtailment = [0.0 for _ in range(time_len)]

    for t in range(td, tfr + 1):

        # 🔸 Step 1：計算此時間步的可用多餘功率（AC側）
        surplus = -T[t] if T[t] < 0 else 0.0

        # 🔸 Step 2：依序讓每顆電池充電
        for i in range(q):

            if t == 0:
                B_prev_i = B_prev[i][t]
            else:
                B_prev_i = B_prev[i][t - 1]

            # 已滿或無剩餘功率則跳過
            if surplus <= 0 or B_prev_i >= B_max[i]:
                continue

            # 本時間步最大可再充多少（AC側轉DC時要乘 eta_c）
            capacity_limit = (B_max[i] - B_prev_i) / max(eta_c, 1e-9)

            # 額定充電功率限制
            rating_limit = P_bat[i]

            # 實際可充功率（AC側）
            charge_power = min(surplus, rating_limit, capacity_limit)

            if charge_power <= 0:
                continue

            P_charge[i][t] = charge_power

            # 從這個時間步的 surplus 扣掉
            surplus -= charge_power

        # 🔸 Step 3：剩下的 surplus → 棄電（不能跨時間）
        curtailment[t] = surplus

    return P_charge, curtailment


# ====================================================
# 1. 資料結構定義（DER / 成本 / 情境 等）
# ====================================================

@dataclass
class CostParameters:
    """每一種 DER 的成本與壽命（給 cost model 用）"""
    # 投資成本 I_i [$]
    I_WT: List[float]
    I_PV: List[float]
    I_DG: List[float]
    I_BAT: List[float]

    # 年度維護成本 (已反映 maintenance level: medium, low, none) Myi [$]
    M_WT: List[float]
    M_PV: List[float]
    M_DG: List[float]
    M_BAT: List[float]

    # 殘值 H_i [$]（在規劃期末）
    H_WT: List[float]
    H_PV: List[float]
    H_DG: List[float]
    H_BAT: List[float]

    # 柴油年耗油成本 F_yi [$]（可由模擬每年總油耗後 f[$/gal]*g[gal/kWh]*Ey 計算）
    F_DG: List[float]

    # 規劃期/貼現
    planning_horizon_years: int  # p
    wacc: float  # r


@dataclass
class MicrogridDesign:
    """微電網架構（設計變數）"""
    # 各 DER 額定功率 [kW]
    P_WT: List[float]      # 風機
    P_PV: List[float]      # 太陽能
    P_DG: List[float]      # 柴油機
    P_BAT: List[float]     # 電池充放功率 [kW]

    # 電池容量 [kWh]
    B_max: List[float]

    # 電池初始 SOC [kWh]
    B_init: List[float]

    # 效率
    eta_c: float           # 充電效率
    eta_d: float           # 放電效率

    # 風機/太陽能容量因子、可用率（可以用常數或時間序列）
    C_WT: float            # 平均容量因子
    C_PV: float            # 平均容量因子
    A_WT: float            # 可用率（例如 0.98）

    # 柴油機參數
    fuel_rate_max: float   # W [gal/hr] 柴油機滿載每小時最大耗油量
    fuel_storage: float    # S [gal] 總油箱容量
    DG_loading: float      # 固定 loading factor （簡化: 0.3~0.8）


@dataclass
class DisturbanceScenario:
    """擾動情境（對應論文的 Sk 與 td, tfr）"""
    name: str
    disturbance_start: int  # td (time index)
    disturbance_end: int    # tfr (time index)
    # 各 DER 的損壞機率 P(d | Sk)
    p_damage_WT: float
    p_damage_PV: float
    p_damage_DG: float
    p_damage_BAT: float
    # 各 DER 的 MTTR [小時]（可從論文表格或 Excel 輸入）
    MTTR_WT: float
    MTTR_PV: float
    MTTR_DG: float
    MTTR_BAT: float


@dataclass
class TimeSeriesInput:
    """時間序列輸入"""
    demand: List[float]  # D_t [kW]
    # 若你有逐時容量因子，可在這裡給完整序列，否則留 None 用平均值
    cf_WT: Optional[List[float]] = None
    cf_PV: Optional[List[float]] = None

    # 若要簡單處理「日夜」效應，可用 hour_of_day 來決定 PV 是否為 0
    hours: Optional[List[int]] = None  # 對照每個 time step 的「小時」(0~23)


@dataclass
class SimulationResult:
    """韌性模擬結果"""
    invulnerability: float
    recovery: float
    resilience: float

    # 時間序列結果
    Pt: List[float]            # microgrid power rating at t
    Gt: List[float]            # delivered power
    Tt: List[float]            # unmet demand (D_t - G_t)
    demand: List[float]

    # 電池相關
    B: List[List[float]]       # SOC [kWh] for each battery over time
    P_charge: List[List[float]]
    P_discharge: List[List[float]]
    curtailment: List[float]

    # 柴油機燃料消耗
    fuel_used: float           # 模擬期間總燃料 [gal]

# ====================================================
# 2. 韌性模擬主流程
# ====================================================

def simulate_microgrid_resilience(
    design: MicrogridDesign,
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    random_seed: Optional[int] = None,
) -> SimulationResult:
    """
    使用論文的韌性定義與流程，
    模擬一個微電網在特定擾動情境下的行為，並計算：
      - invulnerability
      - recovery
      - resilience

    簡化假設：
      - WT / PV 只用容量因子乘上額定功率
      - DG 以固定 loading factor 運轉（有油、有可用時）
      - 電池：先看 WT + PV + DG 是否 > D_t：
            * 若有多餘功率 → 充電（不能跨時間）
            * 若不足 → 放電補足，若仍不足 → T_t 為未滿足需求
      - 損壞與修復：在 td 發生一次損壞判定，修復時間由 MTTR 決定。
    """
    if random_seed is not None:
        random.seed(random_seed)

    D = time_input.demand
    T_len = len(D)

    # 方便起見轉成長度
    n_WT = len(design.P_WT)
    n_PV = len(design.P_PV)
    n_DG = len(design.P_DG)
    n_BAT = len(design.P_BAT)


    # --------------------------
    # 2.1 決定擾動造成的損壞與修復時間
    # --------------------------
    # Ui = 1 → 未損壞；0 → 損壞
    U_WT = [1] * n_WT
    U_PV = [1] * n_PV
    U_DG = [1] * n_DG
    U_BAT = [1] * n_BAT

    # 修復時間（整數 index）
    repair_WT = [None] * n_WT
    repair_PV = [None] * n_PV
    repair_DG = [None] * n_DG
    repair_BAT = [None] * n_BAT

    td = scenario.disturbance_start
    tfr = scenario.disturbance_end

    # 在擾動發生時刻判定損壞（一次）
    for i in range(n_WT):
        if random.random() < scenario.p_damage_WT:
            U_WT[i] = 0
            repair_WT[i] = td + math.ceil(scenario.MTTR_WT)
    for i in range(n_PV):
        if random.random() < scenario.p_damage_PV:
            U_PV[i] = 0
            repair_PV[i] = td + math.ceil(scenario.MTTR_PV)
    for i in range(n_DG):
        if random.random() < scenario.p_damage_DG:
            U_DG[i] = 0
            repair_DG[i] = td + math.ceil(scenario.MTTR_DG)
    for i in range(n_BAT):
        if random.random() < scenario.p_damage_BAT:
            U_BAT[i] = 0
            repair_BAT[i] = td + math.ceil(scenario.MTTR_BAT)

    # --------------------------
    # 2.2 建立可用狀態 O_ti（有損壞、但修復後恢復為 1）
    # --------------------------
    def build_availability(U, repair_list, n_units):
        O = [[1] * T_len for _ in range(n_units)]
        for i in range(n_units):
            if U[i] == 1:
                continue  # 永遠可用
            # 壞掉直到修復
            for t in range(td, min(T_len, (repair_list[i] or T_len))):
                O[i][t] = 0
        return O

    O_WT = build_availability(U_WT, repair_WT, n_WT)
    O_PV = build_availability(U_PV, repair_PV, n_PV)
    O_DG = build_availability(U_DG, repair_DG, n_DG)
    O_BAT = build_availability(U_BAT, repair_BAT, n_BAT)

    # --------------------------
    # 2.3 電池 SOC 與充放電功率初始化
    # --------------------------
    B = [[0.0] * T_len for _ in range(n_BAT)]
    for i in range(n_BAT):
        B[i][0] = design.B_init[i]

    P_charge = [[0.0] * T_len for _ in range(n_BAT)]
    P_discharge = [[0.0] * T_len for _ in range(n_BAT)]
    curtailment = [0.0] * T_len

    # --------------------------
    # 2.4 WT / PV 容量因子時間序列
    # --------------------------
    if time_input.cf_WT is not None:
        cf_WT_ts = time_input.cf_WT
    else:
        cf_WT_ts = [design.C_WT] * T_len

    if time_input.cf_PV is not None:
        cf_PV_ts = time_input.cf_PV
    else:
        # 沒有給逐時 CF 的話，可用簡單日夜模式：
        cf_PV_ts = []
        for t in range(T_len):
            if time_input.hours is not None:
                h = time_input.hours[t]
            else:
                h = t % 24
            # 08:00~20:00 有日照，其他時間 0
            if 8 <= h < 20:
                cf_PV_ts.append(design.C_PV)
            else:
                cf_PV_ts.append(0.0)

    # --------------------------
    # 2.5 逐時模擬功率、燃料、SOC
    # --------------------------
    Pt = [0.0] * T_len   # microgrid power rating
    Gt = [0.0] * T_len   # delivered power
    Tt = [0.0] * T_len   # unmet demand (D_t - G_t)

    fuel_used = 0.0      # 總燃料用量 [gal]
    fuel_remaining = design.fuel_storage

    DG_L = design.DG_loading  # 同一 loading factor（簡化）

    for t in range(T_len):
        Dt = D[t]

        # ---- WT power rating ----
        P_wt_t = 0.0
        for i in range(n_WT):
            if O_WT[i][t] == 1:
                P_wt_t += design.P_WT[i] * cf_WT_ts[t] * design.A_WT

        # ---- PV power rating ----
        P_pv_t = 0.0
        for i in range(n_PV):
            if O_PV[i][t] == 1:
                P_pv_t += design.P_PV[i] * cf_PV_ts[t]

        # ---- DG power rating ----
        P_dg_t = 0.0
        # 本時間步 DG 可用的最大燃料量 [gal/hr]
        if fuel_remaining > 0:
            for i in range(n_DG):
                if O_DG[i][t] == 1:
                    # 額定輸出 * loading factor
                    P_dg_i = design.P_DG[i] * DG_L
                    # 對應燃料消耗
                    fuel_need = design.fuel_rate_max * DG_L  # [gal/hr]
                    # 如果油不夠，就按比例調降
                    if fuel_remaining < fuel_need:
                        ratio = fuel_remaining / fuel_need
                        P_dg_i *= ratio
                        fuel_need = fuel_remaining
                    P_dg_t += P_dg_i
                    fuel_used += fuel_need
                    fuel_remaining -= fuel_need
                    if fuel_remaining <= 0:
                        break
        else:
            P_dg_t = 0.0  # 沒油就不能發電

        # ---- 電池額定放電功率（只算 rating，不算實際放電）----
        # 實際放電功率在「不足」情況時才會用到
        # 若要把電池功率也算進 Pt，可以加總 (但 G_t 不一定用滿)
        P_bat_rating_t = 0.0
        for i in range(n_BAT):
            if O_BAT[i][t] == 1:
                P_bat_rating_t += design.P_BAT[i]

        # ---- microgrid power rating ----
        Pt[t] = P_wt_t + P_pv_t + P_dg_t + P_bat_rating_t

        # ---- 決定有沒有剩餘 / 缺電 (先不考慮電池) ----
        # 對照論文 T_t = D_t - P_t，但這裡先用「不含電池」的版本
        # 供充電決策用
        T_without_batt = Dt - (P_wt_t + P_pv_t + P_dg_t)

        # 取得上一時刻 SOC
        if t == 0:
            B_prev = [design.B_init[i] for i in range(n_BAT)]
        else:
            B_prev = [B[i][t-1] for i in range(n_BAT)]

        # 預設先把 SOC 複製過來，等下再依充放電更新
        for i in range(n_BAT):
            B[i][t] = B_prev[i]

        # ---- Case A: 有剩餘功率 → 充電 ----
        if T_without_batt < 0:
            surplus = -T_without_batt
            # 逐顆電池依序充電（實作上等價於你給的 calculate_charge_power_physical_v1）
            for i in range(n_BAT):
                if O_BAT[i][t] == 0:
                    continue
                if surplus <= 0:
                    break
                if B_prev[i] >= design.B_max[i]:
                    continue

                # 本時間步最大可再充多少（AC 側）
                capacity_limit = (design.B_max[i] - B_prev[i]) / max(design.eta_c, 1e-9)
                rating_limit = design.P_BAT[i]

                charge_power = min(surplus, rating_limit, capacity_limit)
                if charge_power <= 0:
                    continue

                P_charge[i][t] = charge_power
                # SOC 增加的是 DC 側能量：P_charge * eta_c * Δt（1 小時）
                B[i][t] = B_prev[i] + charge_power * design.eta_c
                surplus -= charge_power

            curtailment[t] = surplus  # 剩下的用不掉 → 棄電
            # 因為使用的是「充電用的剩餘」，實際輸送負載仍是 Dt（假設需求都被滿足）
            Gt[t] = Dt
            Tt[t] = 0.0

        # ---- Case B: 功率不足 → 電池放電 ----
        elif T_without_batt > 0:
            deficit = T_without_batt
            supplied_by_batt = 0.0

            for i in range(n_BAT):
                if O_BAT[i][t] == 0:
                    continue
                if deficit <= 0:
                    break
                if B_prev[i] <= 0:
                    continue

                # 最大可輸出的能量（1 小時）= SOC * eta_d
                max_energy_output = B_prev[i] * design.eta_d
                rating_limit = design.P_BAT[i]

                discharge_power = min(deficit, rating_limit, max_energy_output)
                if discharge_power <= 0:
                    continue

                P_discharge[i][t] = discharge_power
                supplied_by_batt += discharge_power
                deficit -= discharge_power
                # 釋出的 AC 能量 = DC 能量 * eta_d
                # DC SOC 消耗 = discharge_power / eta_d
                B[i][t] = max(0.0, B_prev[i] - discharge_power / max(design.eta_d, 1e-9))

            # 電池補充後的實際供電
            Gt[t] = P_wt_t + P_pv_t + P_dg_t + supplied_by_batt
            # 未滿足負載
            Tt[t] = max(0.0, Dt - Gt[t])

        # ---- Case C: 剛好平衡，無需電池 ----
        else:
            Gt[t] = Dt
            Tt[t] = 0.0
            curtailment[t] = 0.0

    # ====================================================
    # 2.6 計算韌性指標 invulnerability / recovery / resilience
    # ====================================================

    # invulnerability = P_ts / P_td
    # P_td = 擾動發生時 (td) 的 microgrid power rating
    P_td = Pt[scenario.disturbance_start]
    # P_ts = 穩定後的 power rating（這裡簡化取 tfr 之後第一個時間點）
    P_ts = Pt[min(scenario.disturbance_end, T_len - 1)]

    if P_td <= 0:
        invulnerability = 0.0
    else:
        invulnerability = P_ts / P_td

    # recovery = 1 - sum_{t=td}^{tfr} (D_t - G_t) / sum_{t=td}^{tfr} D_t
    num = 0.0
    den = 0.0
    for t in range(scenario.disturbance_start, min(scenario.disturbance_end + 1, T_len)):
        num += max(0.0, D[t] - Gt[t])  # unmet demand
        den += D[t]

    if den <= 0:
        recovery = 0.0
    else:
        recovery = 1.0 - num / den

    resilience = 0.5 * (invulnerability + recovery)

    return SimulationResult(
        invulnerability=invulnerability,
        recovery=recovery,
        resilience=resilience,
        Pt=Pt,
        Gt=Gt,
        Tt=Tt,
        demand=D,
        B=B,
        P_charge=P_charge,
        P_discharge=P_discharge,
        curtailment=curtailment,
        fuel_used=fuel_used,
    )


# ====================================================
# 3. 成本模型：LCOE 與 LCOED
# ====================================================

def _npv_series(values: List[float], r: float) -> float:
    """折現一個年度現金流序列（從 year=1 開始）"""
    return sum(v / ((1 + r) ** (y + 1)) for y, v in enumerate(values))


def compute_LCOE(cost: CostParameters, Ey_year: List[float]) -> float:
    """
    對應論文式 (3.1)，做「合理化」版本：

    LCOE = NPV(投資 + 維護 + 燃料 - 殘值) / NPV(發電量 Ey)

    這裡假設：
      - I_WT / I_PV / I_DG / I_BAT (投資) 全在第 1 年發生
      - 殘值 H_* 在規劃期末一次性回收
      - Myi, Fyi 每年重複（長度需 >= planning_horizon_years）
    """
    r = cost.wacc
    p = cost.planning_horizon_years

    # 投資成本總額（year 1，一次性）
    I_total = (
        sum(cost.I_WT) + sum(cost.I_PV) +
        sum(cost.I_DG) + sum(cost.I_BAT)
    )
    # 殘值總額（規劃期末一次）
    H_total = (
        sum(cost.H_WT) + sum(cost.H_PV) +
        sum(cost.H_DG) + sum(cost.H_BAT)
    )

    # 每年的 M + F
    # 若給的序列比規劃期短，則用最後一年延伸
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
    F_DG_y = pad(cost.F_DG)
    Ey_y = pad(Ey_year)

    annual_costs = []
    for y in range(p):
        # 第一年加上投資成本
        invest = I_total if y == 0 else 0.0
        # 第 p 年減掉殘值
        salvage = H_total if y == p - 1 else 0.0
        annual = (
            invest
            + M_WT_y[y] + M_PV_y[y] + M_DG_y[y] + M_BAT_y[y]
            + F_DG_y[y]
            - salvage
        )
        annual_costs.append(annual)

    numerator = _npv_series(annual_costs, r)
    denominator = _npv_series(Ey_y, r)

    if denominator <= 0:
        return float("inf")
    return numerator / denominator


def compute_LCOED(cost: CostParameters, Dy_year: List[float]) -> float:
    """
    對應論文式 (3.4) 的 LCOED（Life Cycle Cost of Energy for Demand）

    LCOED = NPV(風 + 光 + 柴 + 電池之 投資+維護+燃料 - 殘值) / NPV(需求 Dy)

    這裡也採用標準 NPV 寫法：
      - 投資成本在第 1 年
      - 殘值在規劃期末
      - M, F 每年發生
    """
    r = cost.wacc
    p = cost.planning_horizon_years

    def pad(lst: List[float]) -> List[float]:
        if len(lst) >= p:
            return lst[:p]
        if not lst:
            return [0.0] * p
        return lst + [lst[-1]] * (p - len(lst))

    Dy_y = pad(Dy_year)

    I_total = (
        sum(cost.I_WT) + sum(cost.I_PV) +
        sum(cost.I_DG) + sum(cost.I_BAT)
    )
    H_total = (
        sum(cost.H_WT) + sum(cost.H_PV) +
        sum(cost.H_DG) + sum(cost.H_BAT)
    )

    M_WT_y = pad(cost.M_WT)
    M_PV_y = pad(cost.M_PV)
    M_DG_y = pad(cost.M_DG)
    M_BAT_y = pad(cost.M_BAT)
    F_DG_y = pad(cost.F_DG)

    annual_costs = []
    for y in range(p):
        invest = I_total if y == 0 else 0.0
        salvage = H_total if y == p - 1 else 0.0
        annual = (
            invest
            + M_WT_y[y] + M_PV_y[y] + M_DG_y[y] + M_BAT_y[y]
            + F_DG_y[y]
            - salvage
        )
        annual_costs.append(annual)

    numerator = _npv_series(annual_costs, r)
    denominator = _npv_series(Dy_y, r)

    if denominator <= 0:
        return float("inf")
    return numerator / denominator


# ====================================================
# 4. 簡單設計搜尋：多組架構 → 韌性 + 成本
# ====================================================

def evaluate_designs(
    designs: List[MicrogridDesign],
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    base_cost: CostParameters,
    scale_to_year: float = 365.0,  # 若模擬 24h，可乘 365 變成年用電
    random_seed: Optional[int] = None,
) -> List[Dict]:
    """
    對多組微電網設計進行評估：
      - 先跑一次韌性模擬
      - 用模擬得到的 G_t 與 D_t 推估年發電 / 年需求
      - 用 cost model 算 LCOE / LCOED
      - 回傳列表，供畫 Pareto 曲線或選擇最佳設計

    回傳每組設計的 dict 包含：
      {
        "design_index": k,
        "resilience": ...,
        "invulnerability": ...,
        "recovery": ...,
        "LCOE": ...,
        "LCOED": ...,
        ...
      }
    """
    results = []

    for idx, design in enumerate(designs):
        sim = simulate_microgrid_resilience(
            design=design,
            scenario=scenario,
            time_input=time_input,
            random_seed=(None if random_seed is None else random_seed + idx),
        )

        # 以單次模擬結果推估年度 Ey, Dy（這裡簡單地乘上 scale_to_year）
        Ey_day = sum(sim.Gt)  # kWh，假設 Δt = 1hr
        Dy_day = sum(sim.demand)

        Ey_year = [Ey_day * scale_to_year]  # 若規劃期 > 1 年，cost model 會自動延伸
        Dy_year = [Dy_day * scale_to_year]

        LCOE = compute_LCOE(base_cost, Ey_year)
        LCOED = compute_LCOED(base_cost, Dy_year)

        results.append(
            {
                "design_index": idx,
                "invulnerability": sim.invulnerability,
                "recovery": sim.recovery,
                "resilience": sim.resilience,
                "LCOE": LCOE,
                "LCOED": LCOED,
                "fuel_used": sim.fuel_used,
            }
        )

    return results


# ====================================================
# 5. 簡單使用範例（可自行修改或刪除）
# ====================================================

if __name__ == "__main__":

    # 讀取 Excel：
    df_demand = pd.read_csv("Demand_profile_1D.csv")
    demand_24h = df_demand["D_kW"].values.tolist()

    hours = list(range(0, 23))  # 8~31 對應到 24 小時（mod 24 即可）

    # 讀取 CSV
    df_cf_pv = pd.read_csv("cf_PV_t_1D.csv")
    df_cf_wt = pd.read_csv("cf_WT_t_1D.csv")
    # 若 CSV 裡的順序就是 0~23 小時，直接取用
    cf_WT_24h = df_cf_wt["wt_cf"].tolist()
    cf_PV_24h = df_cf_pv["pv_cf"].tolist()
    
    time_input = TimeSeriesInput(
        demand=demand_24h,
        cf_WT= cf_WT_24h,  
        cf_PV=cf_PV_24h,  
        hours=[h % 24 for h in hours],
    )
    
    # 一個示範用的微電網設計
    design_example = MicrogridDesign(
        P_WT=[2000.0]*5,     # 5 台 2.0 MW 風機
        P_PV=[1000.0]*20,     # 20 組 1 MW 太陽能
        P_DG=[5000.0]*3,     # 3 台 5 MW 柴油機
        P_BAT=[2000.0]*10,    # 10 組 2 MW 電池
        B_max=[4000.0]*10,    # 4 MWh
        B_init=[4000.0]*10,   # 滿電
        eta_c=0.97,
        eta_d=0.98,
        C_WT=0.22,
        C_PV=0.36,
        A_WT=0.98,
        fuel_rate_max=350.0,    # [gal/hr]
        fuel_storage=40000.0,   # [gal]
        DG_loading=0.7,
    )

    scenario_hurricane = DisturbanceScenario(
        name="Hurricane",
        disturbance_start=7,
        disturbance_end=8,
        p_damage_WT=0.5,
        p_damage_PV=0.7,
        p_damage_DG=0.3,
        p_damage_BAT=0.2,
        MTTR_WT=72,
        MTTR_PV=31,
        MTTR_DG=289,
        MTTR_BAT=3,
    )

    # 成本參數（示範數字，請用你 Excel 裡的真實數據取代）
    cost_params = CostParameters(
        I_WT=[81_100_000.0],
        I_PV=[57_500_000.0],
        I_DG=[109_000_000.0],
        I_BAT=[35_000_000.0],
        M_WT=[3_100_000.0],
        M_PV=[860_000.0],       # I_PV*1.5%
        M_DG=[3_300_000.0],     # I_DG*3%
        M_BAT=[700_000.0],       # I_BAT*2%
        H_WT=[24_000_000.0],     # I_WT*0.3
        H_PV=[20_000_000.0],     # I_PV*0.35
        H_DG=[16_000_000.0],     # I_DG*0.15
        H_BAT=[52_000_000],     # I_BAT*0.15
        F_DG=[208_181_400.0],  # 一年燃料成本 350 gal/hr * 24 hr/day * 365 day * 97 $/gal * 0.7 loading
        planning_horizon_years=10,
        wacc=0.05,
    )

    # 跑一次韌性模擬
    sim_result = simulate_microgrid_resilience(
        design=design_example,
        scenario=scenario_hurricane,
        time_input=time_input,
        random_seed=42,
    )

    print("Invulnerability:", sim_result.invulnerability)
    print("Recovery:", sim_result.recovery)
    print("Resilience:", sim_result.resilience)

    # 推估年發電與需求，算 LCOE / LCOED
    Ey_day = sum(sim_result.Gt)
    Dy_day = sum(sim_result.demand)
    Ey_year = [Ey_day * 365]
    Dy_year = [Dy_day * 365]

    LCOE_val = compute_LCOE(cost_params, Ey_year)
    LCOED_val = compute_LCOED(cost_params, Dy_year)

    print("LCOE  [$ / kWh]:", LCOE_val)
    print("LCOED[$ / kWh]:", LCOED_val)
