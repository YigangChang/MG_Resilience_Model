import pandas as pd
from microgrid.models import DisturbanceScenario, MicrogridDesign, EMSPolicy
from microgrid.simulation import simulate_microgrid_resilience
from microgrid.weather_utils import build_time_input, build_hazard_from_weather


def compute_fuel_storage(fuel_rate_max, days_fuel, P_DG):
    return fuel_rate_max * 24 * days_fuel * len(P_DG)


df_weather = pd.read_csv("Morakot_weather_with_demand.csv")
df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
df_weather["timestamp"] = df_weather["datetime"]

demand_series = df_weather["demand"].tolist()
time_input = build_time_input(df_weather, demand_series)
hazard = build_hazard_from_weather(df_weather)

td_ts = pd.Timestamp("2009-08-07 00:00:00")
tfr_ts = pd.Timestamp("2009-08-10 00:00:00")
try:
    disturbance_start = df_weather.index[df_weather["datetime"] == td_ts][0]
    disturbance_end = df_weather.index[df_weather["datetime"] == tfr_ts][0]
except IndexError:
    disturbance_start = (df_weather["datetime"] - td_ts).abs().idxmin()
    disturbance_end = (df_weather["datetime"] - tfr_ts).abs().idxmin()

scenario = DisturbanceScenario(
    name="Morakot_2009",
    disturbance_start=disturbance_start,
    disturbance_end=disturbance_end,
    base_p_damage_WT=0.10,
    base_p_damage_PV=0.05,
    base_p_damage_DG=0.01,
    base_p_damage_BAT=0.01,
    MTTR_WT=72.0,
    MTTR_PV=31.0,
    MTTR_DG=289.0,
    MTTR_BAT=3.0,
    grid_MTTR_hours=48,
    hazard=hazard,
    evaluation_horizon_hours=168,
    annual_occurrence_rate=0.1,
)

B_max_list = [4000.0] * 5
B_init_list = [b * 0.8 for b in B_max_list]
fuel_rate_max = 350.0
days_fuel = 7
P_DG_1 = [5000.0] * 1

design_1 = MicrogridDesign(
    P_WT=[2000.0] * 5,
    P_PV=[1000.0] * 20,
    P_DG=P_DG_1,
    P_BAT=[2000.0] * 5,
    B_max=B_max_list,
    B_init=B_init_list,
    eta_c=0.97,
    eta_d=0.98,
    C_WT=0.22,
    C_PV=0.36,
    A_WT=0.98,
    fuel_rate_max=fuel_rate_max,
    fuel_storage=compute_fuel_storage(fuel_rate_max, days_fuel, P_DG_1),
    DG_min_loading=0.2,
    DG_max_loading=0.8,
    B_min_soc_frac=0.2,
    B_max_soc_frac=0.8,
    C_rate_charge=1.0,
    C_rate_discharge=1.0,
)

ems_policy = EMSPolicy(
    pre_event_hours=24,
    pre_event_target_soc=0.90,
    pre_event_soc_max=0.90,
    dg_start_soc=0.30,
    dg_stop_soc=0.70,
    load_tier_multipliers=[1.0, 0.6, 0.3],
    load_tier_soc_thresholds=[0.5, 0.3],
)

sim = simulate_microgrid_resilience(
    design=design_1,
    scenario=scenario,
    time_input=time_input,
    critical_load_ratio=0.2,
    random_seed=43,
    ems_policy=ems_policy,
)

# island window used by plots
idx_start = max(0, disturbance_start - 24)
idx_end = min(len(sim.demand) - 1, disturbance_start + scenario.evaluation_horizon_hours)

# derived indicators
records = []
for t in range(idx_start, idx_end + 1):
    dg_avail = sim.U_DG_series[0][t] if sim.U_DG_series else 0
    batt_dis = sum(sim.P_discharge[i][t] for i in range(len(sim.P_discharge)))
    deficit_after = sim.demand[t] - (sim.P_wt[t] + sim.P_pv[t] + batt_dis)
    records.append((
        t,
        sim.ems_mode[t],
        dg_avail,
        sim.avg_soc_frac[t],
        deficit_after,
        sim.P_dg[t],
    ))

island = [r for r in records if "island" in r[1]]
avail_island = [r for r in island if r[2] == 1]
need_dg = [r for r in avail_island if (r[4] > 1e-6 and r[3] <= 0.30)]
active_dg = [r for r in island if r[5] > 1e-6]

print("island_hours", len(island))
print("dg_available_in_island", len(avail_island))
print("dg_needed_hours(deficit>0 & soc<=0.30 & avail=1)", len(need_dg))
print("dg_active_hours", len(active_dg))
print("dg_total_energy", round(sum(sim.P_dg), 3))

print("sample_needed_hours(first_10)")
for r in need_dg[:10]:
    t, mode, avail, soc, deficit, pdg = r
    print(t, mode, avail, round(soc, 3), round(deficit, 1), round(pdg, 1))
