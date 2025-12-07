# microgrid/models.py
from dataclasses import dataclass
from typing import List, Optional


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
    grid_MTTR_hours: int = 48   # 預設 48 小時


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
