# microgrid/monte_carlo.py
"""
Monte Carlo Uncertainty Analysis for Microgrid Resilience
評估以下不確定性：
1. 電力設備故障機率（WT、PV、DG、BAT）
2. 故障設備修復時間（MTTR）
3. 柴油發電機可用燃料量
4. 實際燃料消耗率
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import math

from .models import (
    MicrogridDesign,
    DisturbanceScenario,
    TimeSeriesInput,
    EMSPolicy,
)
from .simulation import simulate_microgrid_resilience


@dataclass
class UncertaintyDistribution:
    """設備不確定性參數分佈"""
    name: str
    mean: float
    std_dev: float
    min_val: float = 0.0
    max_val: float = 1.0
    distribution_type: str = "normal"  # "normal", "lognormal", "uniform", "triangular"
    
    def sample(self) -> float:
        """根據分佈類型進行採樣"""
        if self.distribution_type == "normal":
            value = np.random.normal(self.mean, self.std_dev)
        elif self.distribution_type == "lognormal":
            # 對數常態分佈（適合MTTR等正偏態數據）
            value = np.random.lognormal(
                np.log(self.mean), 
                self.std_dev
            )
        elif self.distribution_type == "uniform":
            value = np.random.uniform(self.min_val, self.max_val)
        elif self.distribution_type == "triangular":
            # 三角分佈
            value = np.random.triangular(
                self.min_val, 
                self.mean, 
                self.max_val
            )
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        # 限制在範圍內
        return np.clip(value, self.min_val, self.max_val)


@dataclass
class MonteCarloConfig:
    """Monte Carlo 分析配置"""
    # 設備故障率不確定性（基值 ± std_dev）
    p_fail_WT: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="WT_failure_probability",
            mean=0.05,  # 5% 基值
            std_dev=0.02,
            min_val=0.0,
            max_val=0.20,
            distribution_type="normal"
        )
    )
    p_fail_PV: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="PV_failure_probability",
            mean=0.03,  # 3% 基值
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal"
        )
    )
    p_fail_DG: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="DG_failure_probability",
            mean=0.02,  # 2% 基值
            std_dev=0.01,
            min_val=0.0,
            max_val=0.10,
            distribution_type="normal"
        )
    )
    p_fail_BAT: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="BAT_failure_probability",
            mean=0.04,  # 4% 基值
            std_dev=0.015,
            min_val=0.0,
            max_val=0.15,
            distribution_type="normal"
        )
    )
    
    # 修復時間不確定性（小時，適用對數常態分佈）
    MTTR_WT: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="WT_MTTR",
            mean=72.0,  # 72 小時
            std_dev=36.0,
            min_val=24.0,
            max_val=240.0,
            distribution_type="lognormal"
        )
    )
    MTTR_PV: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="PV_MTTR",
            mean=48.0,  # 48 小時
            std_dev=24.0,
            min_val=12.0,
            max_val=144.0,
            distribution_type="lognormal"
        )
    )
    MTTR_DG: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="DG_MTTR",
            mean=36.0,  # 36 小時
            std_dev=18.0,
            min_val=12.0,
            max_val=120.0,
            distribution_type="lognormal"
        )
    )
    MTTR_BAT: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="BAT_MTTR",
            mean=24.0,  # 24 小時
            std_dev=12.0,
            min_val=6.0,
            max_val=72.0,
            distribution_type="lognormal"
        )
    )
    
    # 燃料相關不確定性
    fuel_storage_uncertainty: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="fuel_storage_capacity",
            mean=1.0,  # 1.0 表示正常容量
            std_dev=0.15,  # ±15% 不確定性
            min_val=0.70,
            max_val=1.30,
            distribution_type="normal"
        )
    )
    fuel_consumption_rate_uncertainty: UncertaintyDistribution = field(
        default_factory=lambda: UncertaintyDistribution(
            name="fuel_consumption_rate",
            mean=1.0,  # 1.0 表示基值消耗率
            std_dev=0.20,  # ±20% 不確定性
            min_val=0.60,
            max_val=1.50,
            distribution_type="normal"
        )
    )
    
    # MC 參數
    num_simulations: int = 1000  # Monte Carlo 試驗次數
    random_seed: Optional[int] = None


@dataclass
class MonteCarloResult:
    """Monte Carlo 分析結果"""
    # 總體性能指標統計
    NPR_mean: float
    NPR_std: float
    NPR_median: float
    NPR_percentile_5: float
    NPR_percentile_95: float
    NPR_samples: List[float]
    
    CLSR_mean: float
    CLSR_std: float
    CLSR_median: float
    CLSR_percentile_5: float
    CLSR_percentile_95: float
    CLSR_samples: List[float]
    
    EID_mean: float
    EID_std: float
    EID_median: float
    EID_percentile_5: float
    EID_percentile_95: float
    EID_samples: List[float]
    
    # 燃料相關統計
    fuel_consumed_mean: float
    fuel_consumed_std: float
    fuel_consumed_max: float
    fuel_consumed_min: float
    fuel_consumed_samples: List[float]
    
    fuel_remaining_mean: float
    fuel_remaining_std: float
    fuel_remaining_min: float
    fuel_remaining_max: float
    fuel_remaining_samples: List[float]
    
    fuel_shortage_probability: float  # 燃料不足的機率
    
    # 故障發生統計
    equipment_failure_stats: Dict[str, Dict[str, float]]  # 各設備故障統計
    
    # 修復時間統計
    repair_time_stats: Dict[str, Dict[str, float]]  # 各設備修復時間統計
    
    def to_dataframe(self) -> pd.DataFrame:
        """將結果轉換為 DataFrame"""
        summary_data = {
            'Metric': [
                'NPR Mean', 'NPR Std', 'NPR Median', 'NPR 5%ile', 'NPR 95%ile',
                'CLSR Mean', 'CLSR Std', 'CLSR Median', 'CLSR 5%ile', 'CLSR 95%ile',
                'EID Mean', 'EID Std', 'EID Median', 'EID 5%ile', 'EID 95%ile',
                'Fuel Consumed Mean (gal)', 'Fuel Consumed Std', 'Fuel Consumed Max',
                'Fuel Remaining Mean (gal)', 'Fuel Shortage Probability'
            ],
            'Value': [
                self.NPR_mean, self.NPR_std, self.NPR_median, self.NPR_percentile_5, self.NPR_percentile_95,
                self.CLSR_mean, self.CLSR_std, self.CLSR_median, self.CLSR_percentile_5, self.CLSR_percentile_95,
                self.EID_mean, self.EID_std, self.EID_median, self.EID_percentile_5, self.EID_percentile_95,
                self.fuel_consumed_mean, self.fuel_consumed_std, self.fuel_consumed_max,
                self.fuel_remaining_mean, self.fuel_shortage_probability
            ]
        }
        return pd.DataFrame(summary_data)


class MonteCarloAnalyzer:
    """蒙地卡羅不確定分析引擎"""
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
    
    def run_analysis(
        self,
        design: MicrogridDesign,
        scenario: DisturbanceScenario,
        time_input: TimeSeriesInput,
        critical_load_ratio: float = 0.2,
        ems_policy: Optional[EMSPolicy] = None,
    ) -> MonteCarloResult:
        """
        執行 Monte Carlo 不確定分析
        
        Parameters:
        -----------
        design : MicrogridDesign
            微電網設計
        scenario : DisturbanceScenario
            災害情景（包含基值故障率和MTTR）
        time_input : TimeSeriesInput
            時間序列輸入（需求、風、光）
        critical_load_ratio : float
            停電時供應的關鍵負載比例
            
        Returns:
        --------
        MonteCarloResult
            包含統計結果的分析結果
        """
        
        NPR_samples = []
        CLSR_samples = []
        EID_samples = []
        fuel_consumed_samples = []
        fuel_remaining_samples = []
        failure_events = {
            'WT': [], 'PV': [], 'DG': [], 'BAT': []
        }
        repair_times = {
            'WT': [], 'PV': [], 'DG': [], 'BAT': []
        }
        fuel_shortage_count = 0
        
        print(f"開始 Monte Carlo 分析（{self.config.num_simulations} 次模擬）...")
        
        for sim_idx in range(self.config.num_simulations):
            if (sim_idx + 1) % 100 == 0:
                print(f"  進度: {sim_idx + 1}/{self.config.num_simulations}")
            
            # === 採樣不確定參數 ===
            sampled_scenario = self._sample_scenario(scenario)
            sampled_design = self._sample_design(design)
            
            # === 執行模擬 ===
            result = simulate_microgrid_resilience(
                design=sampled_design,
                scenario=sampled_scenario,
                time_input=time_input,
                critical_load_ratio=critical_load_ratio,
                random_seed=None,  # 使用全局 RNG 狀態
                ems_policy=ems_policy,
            )
            
            # === 收集結果 ===
            NPR_samples.append(result.NPR)
            CLSR_samples.append(result.CLSR)
            EID_samples.append(result.EID)
            fuel_consumed_samples.append(result.fuel_used)
            fuel_remaining_samples.append(result.fuel_remaining)
            
            # === 記錄故障事件 ===
            n_WT = len(design.P_WT)
            n_PV = len(design.P_PV)
            n_DG = len(design.P_DG)
            n_BAT = len(design.P_BAT)
            
            failure_events['WT'].append(
                sum(1 for u_series in result.U_WT_series 
                    if min(u_series) == 0)
            )
            failure_events['PV'].append(
                sum(1 for u_series in result.U_PV_series 
                    if min(u_series) == 0)
            )
            failure_events['DG'].append(
                sum(1 for u_series in result.U_DG_series 
                    if min(u_series) == 0)
            )
            failure_events['BAT'].append(
                sum(1 for u_series in result.U_BAT_series 
                    if min(u_series) == 0)
            )
            
            # === 記錄修復時間 ===
            repair_times['WT'].append(sampled_scenario.MTTR_WT)
            repair_times['PV'].append(sampled_scenario.MTTR_PV)
            repair_times['DG'].append(sampled_scenario.MTTR_DG)
            repair_times['BAT'].append(sampled_scenario.MTTR_BAT)
            
            # === 檢查燃料是否不足 ===
            if result.fuel_remaining < 0:
                fuel_shortage_count += 1
        
        # === 計算統計量 ===
        def calc_stats(samples):
            return {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                '5percentile': np.percentile(samples, 5),
                '95percentile': np.percentile(samples, 95),
                'min': np.min(samples),
                'max': np.max(samples),
            }
        
        # === 故障統計 ===
        equipment_failure_stats = {
            'WT': {
                'mean_failures_per_sim': np.mean(failure_events['WT']),
                'std_failures': np.std(failure_events['WT']),
                'max_failures': np.max(failure_events['WT']),
                'failure_probability': np.mean([f > 0 for f in failure_events['WT']]),
            },
            'PV': {
                'mean_failures_per_sim': np.mean(failure_events['PV']),
                'std_failures': np.std(failure_events['PV']),
                'max_failures': np.max(failure_events['PV']),
                'failure_probability': np.mean([f > 0 for f in failure_events['PV']]),
            },
            'DG': {
                'mean_failures_per_sim': np.mean(failure_events['DG']),
                'std_failures': np.std(failure_events['DG']),
                'max_failures': np.max(failure_events['DG']),
                'failure_probability': np.mean([f > 0 for f in failure_events['DG']]),
            },
            'BAT': {
                'mean_failures_per_sim': np.mean(failure_events['BAT']),
                'std_failures': np.std(failure_events['BAT']),
                'max_failures': np.max(failure_events['BAT']),
                'failure_probability': np.mean([f > 0 for f in failure_events['BAT']]),
            },
        }
        
        # === 修復時間統計 ===
        repair_time_stats = {
            'WT': calc_stats(repair_times['WT']),
            'PV': calc_stats(repair_times['PV']),
            'DG': calc_stats(repair_times['DG']),
            'BAT': calc_stats(repair_times['BAT']),
        }
        
        # === 建立結果物件 ===
        result = MonteCarloResult(
            # NPR
            NPR_mean=np.mean(NPR_samples),
            NPR_std=np.std(NPR_samples),
            NPR_median=np.median(NPR_samples),
            NPR_percentile_5=np.percentile(NPR_samples, 5),
            NPR_percentile_95=np.percentile(NPR_samples, 95),
            NPR_samples=NPR_samples,
            # CLSR
            CLSR_mean=np.mean(CLSR_samples),
            CLSR_std=np.std(CLSR_samples),
            CLSR_median=np.median(CLSR_samples),
            CLSR_percentile_5=np.percentile(CLSR_samples, 5),
            CLSR_percentile_95=np.percentile(CLSR_samples, 95),
            CLSR_samples=CLSR_samples,
            # EID
            EID_mean=np.mean(EID_samples),
            EID_std=np.std(EID_samples),
            EID_median=np.median(EID_samples),
            EID_percentile_5=np.percentile(EID_samples, 5),
            EID_percentile_95=np.percentile(EID_samples, 95),
            EID_samples=EID_samples,
            # 燃料
            fuel_consumed_mean=np.mean(fuel_consumed_samples),
            fuel_consumed_std=np.std(fuel_consumed_samples),
            fuel_consumed_max=np.max(fuel_consumed_samples),
            fuel_consumed_min=np.min(fuel_consumed_samples),
            fuel_consumed_samples=fuel_consumed_samples,
            fuel_remaining_mean=np.mean(fuel_remaining_samples),
            fuel_remaining_std=np.std(fuel_remaining_samples),
            fuel_remaining_min=np.min(fuel_remaining_samples),
            fuel_remaining_max=np.max(fuel_remaining_samples),
            fuel_remaining_samples=fuel_remaining_samples,
            fuel_shortage_probability=fuel_shortage_count / self.config.num_simulations,
            # 故障統計
            equipment_failure_stats=equipment_failure_stats,
            repair_time_stats=repair_time_stats,
        )
        
        print(f"Monte Carlo 分析完成！")
        print(f"  NPR: {result.NPR_mean:.4f} ± {result.NPR_std:.4f}")
        print(f"  CLSR: {result.CLSR_mean:.4f} ± {result.CLSR_std:.4f}")
        print(f"  EID: {result.EID_mean:.4f} ± {result.EID_std:.4f}")
        print(f"  燃料不足機率: {result.fuel_shortage_probability:.2%}")
        
        return result
    
    def _sample_scenario(self, scenario: DisturbanceScenario) -> DisturbanceScenario:
        """採樣一個具有不確定參數的情景副本"""
        from dataclasses import replace
        
        return replace(
            scenario,
            base_p_damage_WT=self.config.p_fail_WT.sample(),
            base_p_damage_PV=self.config.p_fail_PV.sample(),
            base_p_damage_DG=self.config.p_fail_DG.sample(),
            base_p_damage_BAT=self.config.p_fail_BAT.sample(),
            MTTR_WT=self.config.MTTR_WT.sample(),
            MTTR_PV=self.config.MTTR_PV.sample(),
            MTTR_DG=self.config.MTTR_DG.sample(),
            MTTR_BAT=self.config.MTTR_BAT.sample(),
        )
    
    def _sample_design(self, design: MicrogridDesign) -> MicrogridDesign:
        """採樣一個具有不確定參數的設計副本"""
        from dataclasses import replace
        
        fuel_storage_mult = self.config.fuel_storage_uncertainty.sample()
        fuel_rate_mult = self.config.fuel_consumption_rate_uncertainty.sample()
        
        return replace(
            design,
            fuel_storage=design.fuel_storage * fuel_storage_mult,
            fuel_rate_max=design.fuel_rate_max * fuel_rate_mult,
        )
    
    def generate_report(self, result: MonteCarloResult, output_path: str = None) -> str:
        """生成詳細的 Monte Carlo 分析報告"""
        report = []
        report.append("=" * 80)
        report.append("蒙地卡羅不確定分析報告")
        report.append("=" * 80)
        report.append("")
        
        # 總結統計
        report.append("【韌性性能指標 - 統計摘要】")
        report.append("-" * 80)
        report.append(f"淨初級收益率 (NPR):")
        report.append(f"  平均值:     {result.NPR_mean:.6f}")
        report.append(f"  標準差:     {result.NPR_std:.6f}")
        report.append(f"  中位數:     {result.NPR_median:.6f}")
        report.append(f"  5 百分位:   {result.NPR_percentile_5:.6f}")
        report.append(f"  95 百分位:  {result.NPR_percentile_95:.6f}")
        report.append("")
        
        report.append(f"關鍵負載生存比率 (CLSR):")
        report.append(f"  平均值:     {result.CLSR_mean:.6f}")
        report.append(f"  標準差:     {result.CLSR_std:.6f}")
        report.append(f"  中位數:     {result.CLSR_median:.6f}")
        report.append(f"  5 百分位:   {result.CLSR_percentile_5:.6f}")
        report.append(f"  95 百分位:  {result.CLSR_percentile_95:.6f}")
        report.append("")
        
        report.append(f"能源不足指數 (EID):")
        report.append(f"  平均值:     {result.EID_mean:.6f}")
        report.append(f"  標準差:     {result.EID_std:.6f}")
        report.append(f"  中位數:     {result.EID_median:.6f}")
        report.append(f"  5 百分位:   {result.EID_percentile_5:.6f}")
        report.append(f"  95 百分位:  {result.EID_percentile_95:.6f}")
        report.append("")
        
        # 燃料分析
        report.append("【燃料相關分析】")
        report.append("-" * 80)
        report.append(f"柴油消耗量 (加侖):")
        report.append(f"  平均值:     {result.fuel_consumed_mean:.2f}")
        report.append(f"  標準差:     {result.fuel_consumed_std:.2f}")
        report.append(f"  最小值:     {result.fuel_consumed_min:.2f}")
        report.append(f"  最大值:     {result.fuel_consumed_max:.2f}")
        report.append("")
        
        report.append(f"剩餘燃料量 (加侖):")
        report.append(f"  平均值:     {result.fuel_remaining_mean:.2f}")
        report.append(f"  標準差:     {result.fuel_remaining_std:.2f}")
        report.append(f"  最小值:     {result.fuel_remaining_min:.2f}")
        report.append(f"  最大值:     {result.fuel_remaining_max:.2f}")
        report.append("")
        
        report.append(f"燃料短缺風險:")
        report.append(f"  不足機率:   {result.fuel_shortage_probability:.2%}")
        report.append("")
        
        # 設備故障分析
        report.append("【設備故障分析】")
        report.append("-" * 80)
        for equipment, stats in result.equipment_failure_stats.items():
            report.append(f"{equipment} (Wind Turbine / PV / DG / Battery):")
            report.append(f"  平均每次模擬故障數: {stats['mean_failures_per_sim']:.2f}")
            report.append(f"  故障數標準差:       {stats['std_failures']:.2f}")
            report.append(f"  最大故障數:         {stats['max_failures']}")
            report.append(f"  故障發生機率:       {stats['failure_probability']:.2%}")
            report.append("")
        
        # 修復時間分析
        report.append("【修復時間分析 (小時)】")
        report.append("-" * 80)
        equipment_names = {'WT': '風力機', 'PV': '光伏', 'DG': '柴油機', 'BAT': '電池'}
        for equipment, stats in result.repair_time_stats.items():
            report.append(f"{equipment}（{equipment_names[equipment]}）:")
            report.append(f"  平均修復時間: {stats['mean']:>8.2f} 小時")
            report.append(f"  標準差:       {stats['std']:>8.2f} 小時")
            report.append(f"  中位數:       {stats['median']:>8.2f} 小時")
            report.append(f"  最小值:       {stats['min']:>8.2f} 小時")
            report.append(f"  最大值:       {stats['max']:>8.2f} 小時")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
