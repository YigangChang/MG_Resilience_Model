# microgrid/cost.py
from typing import List, Dict, Optional

from .models import CostParameters, MicrogridDesign, DisturbanceScenario, TimeSeriesInput, EMSPolicy
from .simulation import simulate_microgrid_resilience


# ======================================================
#   Resilience-oriented Economic Metrics (Core)
# ======================================================

def compute_A_EENS(
    EENS_baseline: float,
    EENS_strategy: float,
    VOLL: float,
) -> float:
    """
    Avoided Energy Not Supplied Cost (A-EENS)

    A-EENS = (EENS_baseline - EENS_strategy) * VOLL
    """
    return (EENS_baseline - EENS_strategy) * VOLL


def compute_CPSO(
    C_out: float,
    E_outage_served: float,
) -> Optional[float]:
    """
    Cost per kWh Served During Outage (CPSO)

    CPSO = C_out / E_outage_served
    """
    if E_outage_served <= 0:
        return None
    return C_out / E_outage_served


def compute_CPH(
    C_out: float,
    EID: float,
) -> Optional[float]:
    """
    Cost per Hour of Sustained Critical Load (CPH)

    CPH = C_out / EID
    """
    if EID <= 0:
        return None
    return C_out / EID


def compute_EARC(
    C_fix: float,
    lambda_event: float,
    scenarios: List[Dict],
    VOLL: float,
) -> float:
    """
    Expected Annual Resilience Cost (EARC)

    EARC = C_fix + lambda * sum_s p_s * (EENS_s * VOLL + C_var_s)

    scenarios: List of dicts with keys:
        - p_s: scenario probability
        - EENS: Energy Not Supplied (kWh)
        - C_var: variable cost during outage ($)
    """
    expected_event_cost = sum(
        s["p_s"] * (s["EENS"] * VOLL + s["C_var"])
        for s in scenarios
    )
    return C_fix + lambda_event * expected_event_cost


def compute_RBCR(
    expected_A_EENS: float,
    delta_C_fix: float,
    lambda_event: float,
    scenarios: List[Dict],
) -> Optional[float]:
    """
    Resilience Benefit–Cost Ratio (RBCR)

    RBCR = E[A-EENS] / (ΔC_fix + λ * sum_s p_s * ΔC_var_s)

    scenarios: List of dicts with keys:
        - p_s
        - delta_C_var
    """
    denominator = (
        delta_C_fix
        + lambda_event * sum(
            s["p_s"] * s["delta_C_var"] for s in scenarios
        )
    )
    if denominator <= 0:
        return None
    return expected_A_EENS / denominator


# ======================================================
#   Design-level Economic Evaluation (Single Scenario)
# ======================================================

def evaluate_designs(
    designs: List[MicrogridDesign],
    scenario: DisturbanceScenario,
    time_input: TimeSeriesInput,
    cost: CostParameters,
    days_in_dataset: float,
    critical_load_ratio: float = 0.2,
    random_seed: Optional[int] = None,
    ems_policy: Optional[EMSPolicy] = None,
) -> List[Dict]:
    """
    Evaluate microgrid designs using resilience-oriented economic metrics only.

    Metrics returned per design:
        - EENS
        - CPSO
        - CPH
        - EARC
        - (RBCR calculated outside this function)
    """

    results = []

    VOLL = cost.VOLL
    lambda_event = scenario.annual_occurrence_rate
    C_fix = cost.C_fix_baseline
    fuel_price = cost.fuel_price_per_gal

    for idx, design in enumerate(designs):
        sim = simulate_microgrid_resilience(
            design=design,
            scenario=scenario,
            time_input=time_input,
            critical_load_ratio=critical_load_ratio,
            random_seed=(None if random_seed is None else random_seed + idx),
            ems_policy=ems_policy,
        )

        # --------------------------------------------------
        #   Outage-level quantities
        # --------------------------------------------------

        # Energy served during outage (critical load only)
        E_outage_served = sum(sim.Gt)

        # Variable outage cost (fuel only; fixed cost excluded)
        C_out = sim.fuel_used * fuel_price

        # --------------------------------------------------
        #   Economic metrics
        # --------------------------------------------------

        
        CPSO = compute_CPSO(C_out, E_outage_served)
        CPH = compute_CPH(C_out, sim.EID)

        EARC = compute_EARC(
            C_fix=C_fix,
            lambda_event=lambda_event,
            scenarios=[
                {
                    "p_s": 1.0,        # single scenario
                    "EENS": sim.EENS,
                    "C_var": C_out,
                }
            ],
            VOLL=VOLL,
        )

        results.append(
            {
                "design_index": idx,

                # ---- Resilience outputs ----
                "EENS": sim.EENS,
                "EID": sim.EID,
                'NPR': sim.NPR,
                'CLSR': sim.CLSR,
                "critical_load_survival_time": sim.critical_load_survival_time,
                "fuel_sustainability_h": sim.fuel_sustainability_h,
                "EENS_ratio": sim.EENS_ratio,
            

                # ---- Economic metrics ----
                
                "CPSO": CPSO,
                "CPH": CPH,
                "EARC": EARC,

                # ---- Operation ----
                "fuel_used": sim.fuel_used,
            }
        )

    return results

# ======================================================
#   Baseline vs Strategy Comparison (RBCR)
# ======================================================

def compare_baseline_and_strategies(
    results: List[Dict],
    cost: CostParameters,
    scenario: DisturbanceScenario,
    baseline_index: int = 0,
) -> List[Dict]:
    """
    Compare baseline design with multiple strategies using RBCR.

    Parameters
    ----------
    results : List[Dict]
        Output from evaluate_designs()
    cost : CostParameters
        Economic parameters (VOLL, fuel_price_per_gal, C_fix_baseline, etc.)
    scenario : DisturbanceScenario
        Contains annual_occurrence_rate (lambda)
    baseline_index : int
        Index of baseline design in results list (default = 0)

    Returns
    -------
    comparison_results : List[Dict]
        Each dict contains:
            - baseline_index
            - strategy_index
            - A_EENS
            - delta_C_fix
            - delta_C_var
            - RBCR
    """

    comparison_results = []

    # --------------------------------------------------
    #   Baseline quantities
    # --------------------------------------------------
    baseline = results[baseline_index]

    EENS_baseline = baseline["EENS"]
    C_var_baseline = baseline["fuel_used"] * cost.fuel_price_per_gal
    C_fix_baseline = cost.C_fix_baseline

    VOLL = cost.VOLL
    lambda_event = scenario.annual_occurrence_rate

    # --------------------------------------------------
    #   Loop through strategies
    # --------------------------------------------------
    for i, res in enumerate(results):
        if i == baseline_index:
            continue

        # ---- Strategy quantities ----
        EENS_strategy = res["EENS"]
        C_var_strategy = res["fuel_used"] * cost.fuel_price_per_gal

        # ---- Avoided outage cost ----
        A_EENS = compute_A_EENS(
            EENS_baseline=EENS_baseline,
            EENS_strategy=EENS_strategy,
            VOLL=VOLL,
        )

        # ---- Fixed cost difference ----
        # 預設：若沒有額外定義 strategy 固定成本，則視為與 baseline 相同
        if hasattr(cost, "C_fix_strategy"):
            C_fix_strategy = cost.C_fix_strategy
        else:
            C_fix_strategy = C_fix_baseline

        delta_C_fix = C_fix_strategy - C_fix_baseline
        delta_C_var = C_var_strategy - C_var_baseline

        # ---- RBCR ----
        RBCR = compute_RBCR(
            expected_A_EENS=A_EENS,
            delta_C_fix=delta_C_fix,
            lambda_event=lambda_event,
            scenarios=[
                {
                    "p_s": 1.0,           # single-scenario comparison
                    "delta_C_var": delta_C_var,
                }
            ],
        )

        comparison_results.append(
            {
                "baseline_index": baseline_index,
                "strategy_index": i,

                # ---- Benefit ----
                "A_EENS": A_EENS,

                # ---- Cost differences ----
                "delta_C_fix": delta_C_fix,
                "delta_C_var": delta_C_var,

                # ---- Decision metric ----
                "RBCR": RBCR,
            }
        )

    return comparison_results
