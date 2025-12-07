# microgrid/cost.py
from typing import List, Dict, Optional

from .models import CostParameters, MicrogridDesign, DisturbanceScenario, TimeSeriesInput
from .simulation import simulate_microgrid_resilience


def _npv_series(values: List[float], r: float) -> float:
    return sum(v / ((1 + r) ** (y + 1)) for y, v in enumerate(values))


def compute_LCOE(
    cost: CostParameters,
    Ey_year: List[float],
    fuel_used_sim: float,
    days_simulated: float = 214.0,
    fuel_price_per_gal: float = 97.0,
) -> float:
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

        LCOE_val = compute_LCOE(
            cost=base_cost,
            Ey_year=Ey_year,
            fuel_used_sim=sim.fuel_used,
            days_simulated=days_in_dataset,
            fuel_price_per_gal=97.0,
        )
        LCOED_val = compute_LCOED(
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
                "LCOE": LCOE_val,
                "LCOED": LCOED_val,
                "fuel_used": sim.fuel_used,
            }
        )

    return results

