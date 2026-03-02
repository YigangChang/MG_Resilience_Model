"""
簡單調試：在 run_simulation.py 執行後，輸出 Design 1 在 8 月 9-10 日的電池放電資料
"""
import sys
sys.path.insert(0, '/Users/Yifang/Desktop/MG_Design_code')

# 直接執行 run_simulation.py 的核心邏輯
exec(open('run_simulation.py').read())

# ============================================================
# 提取 Design 1 的結果並輸出 8 月 9-10 日資料
# ============================================================
if len(sim_results) > 1:
    print("\n" + "=" * 100)
    print("【調試：Design 1 在 8 月 9-10 日的電池放電逐時資料】")
    print("=" * 100)
    
    sim_result_d1 = sim_results[1]  # Design 1
    
    # 找出 8 月 9 日到 8 月 10 日的時間索引
    aug_9_start = "2009-08-09 00:00:00"
    aug_10_end = "2009-08-10 23:00:00"
    
    idx_list = []
    for t in range(len(df_weather)):
        ts = df_weather['datetime'].iloc[t]
        if '2009-08-09' in str(ts) or '2009-08-10' in str(ts):
            idx_list.append(t)
    
    if idx_list:
        idx_start, idx_end = idx_list[0], idx_list[-1]
        print(f"\n時間範圍: {df_weather['datetime'].iloc[idx_start]} 到 {df_weather['datetime'].iloc[idx_end]}")
        print(f"時間索引: t={idx_start} 到 t={idx_end} (共 {idx_end - idx_start + 1} 小時)\n")
        
        print(f"{'時間':<20} {'WT':>8} {'PV':>8} {'DG':>8} {'放電':>8} {'需求':>8} {'供應':>8} {'缺電':>8} {'SOC':>6} {'模式':<20}")
        print("-" * 120)
        
        total_discharge = 0.0
        for t in range(idx_start, idx_end + 1):
            time_str = str(df_weather['datetime'].iloc[t])[:16]
            p_wt = sim_result_d1.P_wt[t]
            p_pv = sim_result_d1.P_pv[t]
            p_dg = sim_result_d1.P_dg[t]
            p_discharge = sum(sim_result_d1.P_discharge[i][t] for i in range(len(sim_result_d1.P_discharge)))
            demand = sim_result_d1.demand[t]
            served = sim_result_d1.Gt[t]
            unserved = sim_result_d1.Tt[t]
            avg_soc = sim_result_d1.avg_soc_frac[t]
            ems_mode = sim_result_d1.ems_mode[t]
            
            total_discharge += p_discharge
            
            print(f"{time_str:<20} {p_wt:>8.0f} {p_pv:>8.0f} {p_dg:>8.0f} {p_discharge:>8.1f} {demand:>8.1f} {served:>8.1f} {unserved:>8.1f} {avg_soc:>6.3f} {ems_mode:<20}")
        
        print("-" * 120)
        print(f"\n電池放電統計 (8月9-10日):")
        print(f"  總放電量: {total_discharge:.1f} kWh")
        print(f"  平均每小時: {total_discharge / (idx_end - idx_start + 1):.1f} kW")
else:
    print("無 Design 1 數據")
