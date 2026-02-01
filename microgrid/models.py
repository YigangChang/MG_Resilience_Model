# microgrid/models.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MicrogridDesign:
    """微電網設備配置 + 運轉限制"""
    P_WT: List[float] #rated power (List of capacities)
    P_PV: List[float]
    P_DG: List[float]
    P_BAT: List[float]

    B_max: List[float] #Maximum battery capacity (kWh)
    B_init: List[float] #Initial battery state of charge (kWh)

    eta_c: float #charging efficiency (0~1)
    eta_d: float #discharging efficiency (0~1)

    C_WT: float #Wind turbine capacity factor (0~1)
    C_PV: float
    A_WT: float #Wind turbine rotor area (m²)??

    fuel_rate_max: float #Maximum fuel consumption rate (gal/h)
    fuel_storage: float #Fuel storage capacity (gal)

    DG_min_loading: float #Minimum diesel generator loading fraction (0~1)
    DG_max_loading: float

    B_min_soc_frac: float   # Battery minimum SOC fraction in normal times 非災害一般下限（例如 0.2）
    B_max_soc_frac: float   # 非災害一般上限（例如 0.8）
    C_rate_charge: float  #C-rate for charging (e.g., 0.5C means 2-hour charge)
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
    base_p_damage_WT: float #Probability of Wind Turbine damage (0~1)
    base_p_damage_PV: float
    base_p_damage_DG: float
    base_p_damage_BAT: float
    MTTR_WT: float #Mean Time to Repair Wind Turbine (hours)
    MTTR_PV: float
    MTTR_DG: float
    MTTR_BAT: float
    hazard: HazardProfile
    evaluation_horizon_hours: int = 168  # Resiliency Curve 評估窗（預設 7 天）
    grid_MTTR_hours: int = 48   # 預設 48 小時 Grid reconnection time
    annual_occurrence_rate: float = 0.1  # 預設每年發生次數

@dataclass
class TimeSeriesInput:
    demand: List[float]
    cf_WT: List[float] #Wind capacity factor time series (0~1)
    cf_PV: List[float]
    hours: Optional[List[int]] = None  # 0~23


@dataclass
class SimulationResult:
    NPR: float
    CLSR: float
    EID: float
   
    Pt: List[float] # total power generation at time t 微電網設備的總容量或總輸出功率，包括 WT、PV、DG 和電池
    Gt: List[float] #實際供應給負載的電力，時間序列記錄
    Tt: List[float] #unserved energy at time t 未供應的電力需求，時間序列記錄
    demand: List[float]  # 這裡用「有效 demand」（災害期間是 critical load）

    B: List[List[float]] #Battery state of charge for each battery unit (kWh)?
    P_charge: List[List[float]] #Charging power for each battery unit (kW)
    P_discharge: List[List[float]]
    curtailment: List[float]

    fuel_used: float
    service_level: List[float] #Load met ratio at each hour (0~1)

    P_wt: List[float]
    P_pv: List[float]
    P_dg: List[float] #	Diesel generator power output

    U_WT_series: List[List[int]] #Wind turbine availability (0=fail, 1=available)
    U_PV_series: List[List[int]]
    U_DG_series: List[List[int]]
    U_BAT_series: List[List[int]]

    # === 新增四項韌性指標 ===
    EENS: float #Expected Energy Not Supplied (kWh)
    LOLE: int #Loss of Load Event (hours)
    critical_load_survival_time: int
    fuel_sustainability_h: float
    EENS_ratio: float #EENS / total demand ratio

@dataclass
class CostParameters:
    I_WT: List[float] #Capital investment cost ($)
    I_PV: List[float]
    I_DG: List[float]
    I_BAT: List[float]

    M_WT: List[float] #Annual fixed O&M cost ($/year)
    M_PV: List[float]
    M_DG: List[float]
    M_BAT: List[float]

    H_WT: List[float] #Salvage/residual value at end of planning horizon ($)
    H_PV: List[float]
    H_DG: List[float]
    H_BAT: List[float]

    VOLL: float #Value of Lost Load ($/kWh)
    C_fix_baseline: float #Baseline fixed cost ($/year)
    C_fix_strategy: float    # $/year (battery + DG annualized)
    fuel_price_per_gal: float #Fuel price ($/gal)
    #planning_horizon_years: int
    #wacc: float #折現率
