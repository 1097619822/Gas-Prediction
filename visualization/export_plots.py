import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 导入 src_v2 核心模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing.data_loader import DataLoader as DL
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.feature_eng import FeatureEngineer as FE
import config

# 学术风格设置
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

def export_academic_plots():
    """
    专门生成用于论文撰写的特征处理过程图。
    """
    plot_dir = os.path.join(config.OUTPUT_DIR, "plots")
    well_files = DL.get_all_well_files()
    target_well = well_files[0] # 选取第一口井作为典型案例
    
    print(f"--- 正在为论文生成学术图表 (案例井: {target_well}) ---")
    
    # 1. 加载原始数据 (包含噪点)
    df_raw = DL.load_well_data(target_well)
    df = DC.unify_formats(df_raw)
    
    # 图 1: 异常值识别诊断图 (3-sigma)
    col = 'gas_volume'
    mean, std = df[col].mean(), df[col].std()
    upper, lower = mean + 3*std, mean - 3*std
    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[col], label='原始产量', color='gray', alpha=0.5)
    plt.axhline(upper, color='red', linestyle='--', label='3σ 上界')
    plt.axhline(lower, color='red', linestyle='--', label='3σ 下界')
    # 标记出异常点
    outliers = df[(df[col] > upper) | (df[col] < lower)]
    plt.scatter(outliers.index, outliers[col], color='red', label='识别出的噪点')
    plt.title(f"图3-1 基于3σ原则的产量异常值识别 ({target_well})")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "01_outlier_diagnosis.png"), dpi=300)
    print(" - 已生成: 01_outlier_diagnosis.png")

    # 2. 数据清洗与插值对比
    df_cleaned = DC.remove_outliers_3sigma(df, [col])
    df_filled = DC.handle_missing_values(df_cleaned, [col])
    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[col], label='原始数据', alpha=0.4)
    plt.plot(df_filled.index, df_filled[col], label='清洗并插值后', color='green', linewidth=1.5)
    plt.title("图3-2 产量数据清洗前后的时序对比")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "02_data_cleaning_compare.png"), dpi=300)
    print(" - 已生成: 02_data_cleaning_compare.png")

    # 3. 特征工程：滑动平均平滑效果
    df_feat = FE.add_rolling_features(df_filled, ['wellhead_press'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_feat.index, df_feat['wellhead_press'], label='原始油压', alpha=0.3)
    plt.plot(df_feat.index, df_feat['wellhead_press_roll_7_mean'], label='7日滑动平均', color='orange')
    plt.title("图3-3 压力特征的滑动平均平滑处理")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "03_feature_smoothing.png"), dpi=300)
    print(" - 已生成: 03_feature_smoothing.png")

    # 4. 特征相关性热力图 (最核心的图)
    df_full = FE.add_lagged_features(df_filled, 'gas_volume')
    cols_for_corr = ['gas_volume', 'wellhead_press', 'gas_volume_lag_1', 'gas_volume_lag_7']
    corr = df_full[cols_for_corr].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("图3-4 产量预测核心特征相关性矩阵")
    plt.savefig(os.path.join(plot_dir, "04_correlation_heatmap.png"), dpi=300)
    print(" - 已生成: 04_correlation_heatmap.png")

    # 5. 产量分布直方图 (数据探索)
    plt.figure(figsize=(8, 5))
    sns.histplot(df_filled['gas_volume'], kde=True, color='skyblue')
    plt.title("图3-5 气井日产量数据频率分布")
    plt.savefig(os.path.join(plot_dir, "05_distribution_hist.png"), dpi=300)
    print(" - 已生成: 05_distribution_hist.png")

    print(f"\n🎉 所有学术图表已保存至: {plot_dir}")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs(os.path.join(config.OUTPUT_DIR, "plots"), exist_ok=True)
    export_academic_plots()
