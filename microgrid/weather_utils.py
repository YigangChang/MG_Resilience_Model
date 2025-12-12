import pandas as pd
import numpy as np
from microgrid.models import TimeSeriesInput, HazardProfile

"""
# =======================================================
# 1. 讀取並合併 7~9 月的 9 個 CSV 變成一個 DataFrame
# =======================================================

def load_three_months_weather(path_wind, path_rain, path_solar):
    df_w = pd.read_csv(path_wind, parse_dates=["datetime"])
    df_r = pd.read_csv(path_rain, parse_dates=["datetime"])
    df_s = pd.read_csv(path_solar, parse_dates=["datetime"])

    # 合併：以 time 為 key
    df = df_w.merge(df_r, on="datetime", how="inner") \
             .merge(df_s, on="datetime", how="inner")

    # 排序
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# =======================================================
# 2. 將 7~9 月資料讀進來
# =======================================================

# 你的檔案名稱（修改成自己的）
df_jul = load_three_months_weather(
    "Morakot_data/wind_200907_pd.csv",
    "Morakot_data/rain_200907_pd.csv",
    "Morakot_data/solar_200907_pd.csv"
)
df_aug = load_three_months_weather(
    "Morakot_data/wind_200908_pd.csv",
    "Morakot_data/rain_200908_pd.csv",
    "Morakot_data/solar_200908_pd.csv"
)
df_sep = load_three_months_weather(
    "Morakot_data/wind_200909_pd.csv",
    "Morakot_data/rain_200909_pd.csv",
    "Morakot_data/solar_200909_pd.csv"
)

# 合併成完整的 2009/07–09 資料
df_weather = pd.concat([df_jul, df_aug, df_sep], axis=0)
df_weather = df_weather.sort_values("datetime").reset_index(drop=True)

print("Loaded Morakot weather data:", df_weather.shape)

print(df_weather.shape)
# =======================================================
# 3. 轉換為容量因子 CF
# =======================================================

def compute_cf_WT(v):
    #Compute wind turbine capacity factor from wind speed.
    cut_in = 3
    rated = 12
    cut_out = 25

    if v < cut_in:
        return 0.0
    if v >= cut_out:
        return 0.0
    if v < rated:
        wind_cf = (v - cut_in) / (rated - cut_in)
        wind_cf = round(wind_cf, 2)
        return wind_cf
    return 1.0

# 風力 CF

df_weather["cf_WT"] = df_weather["wind_speed"].apply(compute_cf_WT)

# PV CF：
# solar 單位：MJ/m² per hour
# 轉成 W/m²
df_weather["irr_Wm2"] = (df_weather["solar"] * 277.7778).round(2)

# 轉成容量因子 CF
df_weather["cf_PV"] = (df_weather["irr_Wm2"] / 1000.0).round(2)

# 確保 CF 在 0~1 之間
df_weather["cf_PV"] = df_weather["cf_PV"].clip(lower=0, upper=1)


# =======================================================
# 4. 產生 hour index（EMS 用）
# =======================================================

df_weather["hour"] = df_weather["datetime"].dt.hour
"""

# =======================================================
# 5. 建立 TimeSeriesInput 物件，替換模型輸入
# =======================================================

def build_time_input(df_weather, demand_series):
    """
    df_weather: 需包含 wind, rain, cf_WT, cf_PV
    demand_series: 小時需求（長度需一致）
    """

    N = len(df_weather)
    if len(demand_series) != N:
        raise ValueError(f"demand_series length {len(demand_series)} != df_weather length {N}")
  

    # 若沒有 time_h（小時），自動產生 0~23 循環
    if "hour" in df_weather.columns:
        time_h = df_weather["hour"].values.tolist()
    else:
        time_h = [(i % 24) for i in range(N)]

    # 若沒有 solar_irradiance，就用 cf_PV * 1000 反推（模型不會真的用它，只是 HazardProfile 需要）
    if "irr_Wm2" in df_weather.columns:
        solar_raw = df_weather["irr_Wm2"].values.tolist()
    else:
        solar_raw = (df_weather["cf_PV"] * 1000).values.tolist()

    ts = TimeSeriesInput(
        cf_WT=df_weather["cf_WT"].values.tolist(),
        cf_PV=df_weather["cf_PV"].values.tolist(),
        demand=demand_series,
        hours=time_h,
    )

    return ts

from microgrid.models import HazardProfile

def build_hazard_from_weather(df_weather: pd.DataFrame) -> HazardProfile:
    """
    從 df_weather 生成 HazardProfile，給 DisturbanceScenario 使用。
    
    需求欄位：
      wind  : 風速 (m/s)
      rain  : 降雨量 (mm/hr)
      solar (或 solar_raw 或 solar_Wm2)：日射量，用來當作 solar_irradiance
    """
    N = len(df_weather)

    # 時間軸：用「模擬小時索引」0,1,2,...,N-1
    time_h = list(range(N))

    # 風速
    if "wind_speed" not in df_weather.columns:
        raise KeyError("df_weather 缺少 'wind_speed' 欄位（m/s）")
    wind_speed = df_weather["wind_speed"].astype(float).tolist()

    # 降雨
    if "rain" not in df_weather.columns:
        raise KeyError("df_weather 缺少 'rain' 欄位（mm/hr）")
    
    #rainfall = pd.to_numeric(df_weather["rain"], errors="coerce").fillna(0.0).tolist()
    # replace "T" which means trace amount with 0.0 in CWA data
    rainfall = df_weather["rain"].astype(float).tolist()

    # 日射量：優先用 solar_Wm2，其次 solar_raw，其次 solar
    if "solar_Wm2" in df_weather.columns:
        solar_irr = df_weather["solar_Wm2"].astype(float).tolist()
    elif "solar_raw" in df_weather.columns:
        solar_irr = df_weather["solar_raw"].astype(float).tolist()
    elif "solar" in df_weather.columns:
        solar_irr = df_weather["solar"].astype(float).tolist()
    else:
        # 沒有的話就先給 0，至少不會壞掉
        solar_irr = [0.0] * N

    hazard = HazardProfile(
        time_h=time_h,
        wind_speed=wind_speed,
        rainfall=rainfall,
        solar_irradiance=solar_irr,
    )
    return hazard

"""
# =======================================================
# 6. 替換你模型使用的 time_input
# =======================================================

import pandas as pd

# 讀取你的 24 小時需求曲線
df_demand = pd.read_csv("Demand_profile_1D.csv")

# 假設欄位名稱叫 "demand_kW" 或只有一欄（你可視狀況調整）
if "D_kW" in df_demand.columns:
    demand_24h = df_demand["D_kW"].tolist()
else:
    # 若只有一欄，取出第一欄
    demand_24h = df_demand.iloc[:, 0].tolist()

# 確保是 24 筆資料
if len(demand_24h) != 24:
    raise ValueError(f"Demand_profile_1D.csv 不是 24 小時資料！共 {len(demand_24h)} 筆")

# 依照莫拉克資料長度展開 demand
N = len(df_weather)  # 例如 2208 小時（92 天）
demand_series = (demand_24h * (N // 24 + 1))[:N]

# 建立 time_input
time_input = build_time_input(df_weather, demand_series)

df_weather["demand"] = demand_series

# 輸出 CSV
output_path = "Morakot_weather_with_demand.csv"
df_weather.to_csv(output_path, index=False, encoding="utf-8-sig")

"""
