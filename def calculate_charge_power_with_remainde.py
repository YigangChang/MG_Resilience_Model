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
