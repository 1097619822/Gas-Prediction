"""
Generate additional thesis figures for multi-well comparison, diagnosis, tuning,
and data preprocessing analysis.

Run from project root:
    python src_v2/generate_extra_figures.py
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "\u8bba\u6587\u56fe\u8868" / "\u8865\u5145\u56fe\u8868"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from pre_processing.data_loader import DataLoader as DL
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.feature_eng import FeatureEngineer as FE


def setup_chinese_font():
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
            return font_name
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    return None


sns.set_theme(style='whitegrid', palette='Set2')
CHINESE_FONT = setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def save_fig(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Saved] {path}")


def load_target_well():
    files = sorted(DL.get_all_well_files())
    if not files:
        raise FileNotFoundError('data目录下没有井数据文件')
    well_file = files[0]
    raw = DL.load_well_data(well_file).sort_values('date').reset_index(drop=True)
    clean = DC.unify_formats(raw)
    clean = DC.remove_outliers_3sigma(clean, ['wellhead_press', 'gas_volume'])
    clean = DC.handle_missing_values(clean, ['wellhead_press', 'gas_volume'])
    fe = FE.add_lagged_features(clean, 'gas_volume')
    fe = FE.add_rolling_features(fe, ['wellhead_press', 'gas_volume']).dropna().reset_index(drop=True)
    return well_file, raw, clean, fe


def plot_multiwell_boxplots():
    df = pd.read_excel(BASE_DIR / 'processed_results' / 'all_well_model_comparison.xlsx')
    order = ['ARIMA', 'SVR', 'Bi-LSTM']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='model', y='RMSE', order=order, ax=ax, width=0.55)
    sns.stripplot(data=df, x='model', y='RMSE', order=order, ax=ax, color='0.25', size=3, alpha=0.55)
    ax.set_title('30口井不同模型RMSE分布箱线图', fontsize=15, fontweight='bold')
    ax.set_xlabel('模型')
    ax.set_ylabel('RMSE')
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, '10_multiwell_rmse_boxplot.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='model', y='R2', order=order, ax=ax, width=0.55)
    sns.stripplot(data=df, x='model', y='R2', order=order, ax=ax, color='0.25', size=3, alpha=0.55)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title('30口井不同模型R²分布箱线图', fontsize=15, fontweight='bold')
    ax.set_xlabel('模型')
    ax.set_ylabel('R²')
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, '11_multiwell_r2_boxplot.png')


def plot_model_win_rate():
    df = pd.read_excel(BASE_DIR / 'processed_results' / 'all_well_model_comparison.xlsx')
    winners = df.loc[df.groupby('well_file')['RMSE'].idxmin(), ['well_file', 'model', 'RMSE']]
    counts = winners['model'].value_counts().reindex(['ARIMA', 'SVR', 'Bi-LSTM'], fill_value=0)
    rates = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=['#6FA8DC', '#93C47D', '#F6B26B'], width=0.55)
    for bar, rate in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f"{int(bar.get_height())}口井\n{rate:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.max() + 3, 5))
    ax.set_title('基于RMSE的多井模型胜率统计', fontsize=15, fontweight='bold')
    ax.set_xlabel('模型')
    ax.set_ylabel('RMSE最优井数')
    ax.grid(axis='y', alpha=0.25)
    save_fig(fig, '12_model_win_rate.png')

    winners.to_excel(OUTPUT_DIR / '12_model_win_rate_detail.xlsx', index=False)


def plot_diagnosis_confusion_matrix():
    cm = pd.read_excel(BASE_DIR / 'processed_results' / 'diagnosis_confusion_matrix.xlsx', index_col=0)
    labels = ['正常生产', '带水生产', '积液风险', '关井']
    cm.index = labels[:len(cm.index)]
    cm.columns = labels[:len(cm.columns)]
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    annot = cm.astype(int).astype(str) + '\n' + cm_norm.round(1).astype(str) + '%'
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', linewidths=0.5,
                cbar_kws={'label': '按真实类别归一化比例(%)'}, ax=ax)
    ax.set_title('积液诊断混淆矩阵热力图', fontsize=15, fontweight='bold')
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    save_fig(fig, '13_diagnosis_confusion_matrix_heatmap.png')


def plot_bilstm_tuning_heatmap():
    df = pd.read_excel(BASE_DIR / 'processed_results' / 'bilstm_tuning_results.xlsx')
    pivot = df.pivot_table(index='seq_len', columns='hidden_size', values='val_RMSE', aggfunc='min')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd_r', linewidths=0.5,
                cbar_kws={'label': '验证集RMSE'}, ax=ax)
    ax.set_title('Bi-LSTM调参结果热力图（RMSE越低越好）', fontsize=15, fontweight='bold')
    ax.set_xlabel('隐藏层维度 hidden_size')
    ax.set_ylabel('输入序列长度 seq_len')
    save_fig(fig, '14_bilstm_tuning_heatmap.png')


def plot_raw_series(raw, well_file):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(raw['date'], raw['gas_volume'], color='#4F81BD', linewidth=1.2)
    ax.set_title(f'{well_file} 原始日产气量时间序列', fontsize=15, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('日产气量')
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    save_fig(fig, '15_raw_gas_volume_series.png')


def plot_cleaning_before_after(raw, clean, well_file):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(raw['date'], raw['gas_volume'], color='#C0504D', linewidth=1.0, label='清洗前')
    axes[0].set_title('清洗前日产气量', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('日产气量')
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(clean['date'], clean['gas_volume'], color='#4F81BD', linewidth=1.0, label='清洗后')
    axes[1].set_title('清洗后日产气量', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('日产气量')
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle(f'{well_file} 数据清洗前后时序对比', fontsize=15, fontweight='bold')
    fig.autofmt_xdate()
    save_fig(fig, '16_cleaning_before_after.png')


def plot_cleaning_stats():
    stats_path = BASE_DIR / 'processed_results' / 'data_cleaning_stats.xlsx'
    stats = pd.read_excel(stats_path)
    row = stats.iloc[0]
    metrics = pd.Series({
        '井口压力缺失': row.get('missing_before_wellhead_press', 0),
        '日产气量缺失': row.get('missing_before_gas_volume', 0),
        '井口压力为0': row.get('zero_wellhead_press', 0),
        '日产气量为0': row.get('zero_gas_volume', 0),
        '井口压力3σ异常': row.get('outliers_wellhead_press_3sigma', 0),
        '日产气量3σ异常': row.get('outliers_gas_volume_3sigma', 0),
        '特征工程丢弃行': row.get('dropped_rows_by_feature_engineering', 0),
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metrics.index, metrics.values, color='#76A5AF')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
    ax.set_title('数据预处理问题统计', fontsize=15, fontweight='bold')
    ax.set_ylabel('记录数')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.25)
    save_fig(fig, '17_preprocessing_issue_stats.png')


def plot_feature_engineering(fe, well_file):
    n = min(len(fe), 180)
    sub = fe.tail(n).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sub['date'], sub['gas_volume'], label='日产气量', linewidth=1.4, color='#4F81BD')
    if 'gas_volume_lag_1' in sub.columns:
        ax.plot(sub['date'], sub['gas_volume_lag_1'], label='滞后1日产气量', linewidth=1.1, alpha=0.8, color='#F6B26B')
    if 'gas_volume_roll_7_mean' in sub.columns:
        ax.plot(sub['date'], sub['gas_volume_roll_7_mean'], label='7日滚动均值', linewidth=1.8, color='#6AA84F')
    ax.set_title(f'{well_file} 特征工程效果示意', fontsize=15, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('日产气量')
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    save_fig(fig, '18_feature_engineering_effect.png')


def plot_train_test_split(fe, well_file):
    split = int(len(fe) * 0.8)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(fe['date'], fe['gas_volume'], color='#4F81BD', linewidth=1.1)
    ax.axvspan(fe['date'].iloc[0], fe['date'].iloc[split - 1], color='#D9EAD3', alpha=0.5, label='训练集')
    ax.axvspan(fe['date'].iloc[split], fe['date'].iloc[-1], color='#FCE5CD', alpha=0.6, label='测试集')
    ax.axvline(fe['date'].iloc[split], color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'{well_file} 训练集与测试集划分示意', fontsize=15, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('日产气量')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    save_fig(fig, '19_train_test_split.png')


def main():
    print('=' * 70)
    print('Generating extra thesis figures')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 70)

    well_file, raw, clean, fe = load_target_well()
    print(f'Using well: {well_file}, raw rows={len(raw)}, cleaned rows={len(clean)}, feature rows={len(fe)}')

    plot_multiwell_boxplots()
    plot_model_win_rate()
    plot_diagnosis_confusion_matrix()
    plot_bilstm_tuning_heatmap()
    plot_raw_series(raw, well_file)
    plot_cleaning_before_after(raw, clean, well_file)
    plot_cleaning_stats()
    plot_feature_engineering(fe, well_file)
    plot_train_test_split(fe, well_file)

    print('=' * 70)
    print('Done')
    print('=' * 70)


if __name__ == '__main__':
    main()
