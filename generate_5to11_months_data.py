import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ===========================================
# 使用者提供的 24 小時資料（作為 daily profile）
# ===========================================
df_demand = pd.read_csv("Demand_profile_1D.csv")
df_pv = pd.read_csv("cf_PV_t_1D.csv")
df_wt = pd.read_csv("cf_WT_t_1D.csv")

demand_24 = df_demand["D_kW"].values
pv_24 = df_pv["pv_cf"].values
wt_24 = df_wt["wt_cf"].values

# ===========================================
# 模擬 5 月 1 日 ~ 11 月 30 日（214 天）
# ===========================================
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 11, 30)
days = (end_date - start_date).days + 1  # 214

total_hours = days * 24

# ===========================================
# 建立時間欄位
# ===========================================
timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]

hour_of_day = [ts.hour for ts in timestamps]
day_index = list(range(total_hours))

# ===========================================
# 建立每日週期負載（含週末下修）
# ===========================================
demand = []
for i, ts in enumerate(timestamps):
    base = demand_24[ts.hour]

    # 週末（六日）降低 10%
    if ts.weekday() >= 5:
        base *= 0.90

    # 夏季高溫（6〜9月）冷卻負載 +5%
    if ts.month in [6, 7, 8, 9] and 13 <= ts.hour <= 16:
        base *= 1.05

    demand.append(base)

# ===========================================
# PV 容量因子（考慮季節性 + 颱風前後降額）
# ===========================================
pv_cf = []
for ts in timestamps:
    base = pv_24[ts.hour]

    # 季節性因子（5月較弱 → 7月最強 → 11月降低）
    seasonal = {
        5: 0.85, 6: 0.95, 7: 1.00, 8: 0.95,
        9: 0.90, 10: 0.80, 11: 0.70
    }[ts.month]

    pv_cf.append(base * seasonal)

# ===========================================
# WT 容量因子（考慮季節風、颱風強風）
# ===========================================
wt_cf = []
for ts in timestamps:
    base = wt_24[ts.hour]

    # 夏秋 (7〜10月) 有較多強風
    seasonal = {
        5: 0.95, 6: 1.00, 7: 1.10, 8: 1.10,
        9: 1.05, 10: 1.00, 11: 0.90
    }[ts.month]

    wt_cf.append(min(base * seasonal, 1.0))

# ===========================================
# Hazard 資料：模擬颱風路徑（風速、降雨）
# 使用歷史台灣強烈颱風的典型分布
# ===========================================
wind_speed = []
rainfall = []
solar_irr = []

# 假設每年 2 次主颱風 (7月 & 9月)，各 48 小時
typhoon_periods = []
typhoon_periods.append((datetime(2024, 7, 23), datetime(2024, 7, 25)))
typhoon_periods.append((datetime(2024, 9, 10), datetime(2024, 9, 12)))

for ts in timestamps:
    in_typhoon = False
    for start, end in typhoon_periods:
        if start <= ts <= end:
            in_typhoon = True
            break

    # 基礎背景風速（非颱風時）
    base_ws = {
        5: 4, 6: 5, 7: 6, 8: 6,
        9: 5, 10: 5, 11: 4
    }[ts.month]

    if in_typhoon:
        # 典型強烈颱風風速分布 (m/s)
        ws = np.random.normal(25, 5)  # 平均 25 m/s
        rain = np.random.normal(60, 20)  # 平均 60 mm/hr
        solar = max(0, pv_24[ts.hour] * 0.1)  # 幾乎沒有日照
    else:
        ws = base_ws + np.random.normal(0, 1)
        rain = max(0, np.random.normal(2, 2))  # 降雨不為負
        solar = pv_24[ts.hour] * 1.0

    wind_speed.append(max(ws, 0))
    rainfall.append(max(rain, 0))
    solar_irr.append(max(min(solar, 1), 0))

# ===========================================
# 輸出 CSV 檔案
# ===========================================
df_out = pd.DataFrame({
    "timestamp": timestamps,
    "day_index": day_index,
    "hour_of_day": hour_of_day,
    "demand_kW": demand,
    "pv_cf": pv_cf,
    "wt_cf": wt_cf,
    "wind_speed_m_s": wind_speed,
    "rain_mm_hr": rainfall,
    "solar_irr_ratio": solar_irr
})

df_out.to_csv("Microgrid_5to11_months_timeseries.csv", index=False)

print("✔ 已輸出 7 個月 (5〜11月) 時間序列資料：")
print("   Microgrid_5to11_months_timeseries.csv")

# ===========================================
#               📊 產生圖表
# ===========================================

# ---- (1) Demand / PV / WT ----
plt.figure(figsize=(14, 5))
plt.plot(df_out["timestamp"], df_out["demand_kW"], label="Demand (kW)")
#plt.plot(df_out["timestamp"], df_out["pv_cf"] * 1000, label="PV CF × 1000", alpha=0.8)
#plt.plot(df_out["timestamp"], df_out["wt_cf"] * 1000, label="WT CF × 1000", alpha=0.8)
plt.title("Demand Time Series (May–Nov)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_demand_pv_wt.png", dpi=200)
plt.show()
"""
# ---- (2) Hazard: Wind speed ----
plt.figure(figsize=(14, 4))
plt.plot(df_out["timestamp"], df_out["wind_speed_m_s"], color="red")
plt.title("Wind Speed (m/s)")
plt.xlabel("Time")
plt.ylabel("Wind Speed")
plt.tight_layout()
plt.savefig("plot_wind_speed.png", dpi=200)
plt.show()

# ---- (3) Hazard: Rainfall ----
plt.figure(figsize=(14, 4))
plt.plot(df_out["timestamp"], df_out["rain_mm_hr"], color="blue")
plt.title("Rainfall (mm/hr)")
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.tight_layout()
plt.savefig("plot_rainfall.png", dpi=200)
plt.show()

# ---- (4) Hazard: Solar irradiance ----
plt.figure(figsize=(14, 4))
plt.plot(df_out["timestamp"], df_out["solar_irr_ratio"], color="orange")
plt.title("Solar Irradiance Ratio")
plt.xlabel("Time")
plt.ylabel("Ratio")
plt.tight_layout()
plt.savefig("plot_solar_irr.png", dpi=200)
plt.show()
"""
print("✔ 已完成所有圖表繪製：")
print("   plot_demand_pv_wt.png")
print("   plot_wind_speed.png")
print("   plot_rainfall.png")
print("   plot_solar_irr.png")