"""
毕业论文图表生成模块
包含：特征热力图、模型架构图、预测对比图、残差分析图、指标对比图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
def setup_chinese_font():
    font_candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\NotoSansSC-VF.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
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


CHINESE_FONT = setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 创建输出目录
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '\u8bba\u6587\u56fe\u8868'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    """保存图片到论文图表目录"""
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{name}.png"))
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[Saved] {path}")
    return path


# ==========================================
# 1. 特征相关性热力图
# ==========================================
def plot_feature_correlation(df, feature_cols, title='特征相关性热力图', save_name='01_feature_correlation'):
    """
    绘制特征相关性热力图

    Args:
        df: DataFrame
        feature_cols: 特征列名列表
        title: 图表标题
        save_name: 保存文件名
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 计算相关性矩阵
    corr = df[feature_cols].corr()

    # 绘制热力图
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 只显示下三角
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot_kws={'size': 10}, ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 添加显著性标注说明
    textstr = '颜色说明:\n深蓝: 强负相关 (-1)\n白色: 无相关 (0)\n深红: 强正相关 (+1)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.25, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)

    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 2. LSTM 模型架构图
# ==========================================
def plot_lstm_architecture(save_name='02_lstm_architecture'):
    """绘制 LSTM 模型架构图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 颜色定义
    colors = {
        'input': '#E8F4FD',      # 浅蓝
        'lstm': '#FFF4E6',       # 浅橙
        'dense': '#E8F8F5',      # 浅绿
        'output': '#F5EEF8',     # 浅紫
        'arrow': '#5D6D7E'       # 灰蓝
    }

    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # 输入层
    draw_box(0.5, 6.5, 2, 1.5, 'Input\n(batch, seq_len, 3)', colors['input'])
    draw_arrow(2.5, 7.25, 3.5, 7.25)

    # LSTM 层 1 (双向)
    draw_box(3.5, 6, 2.5, 2.5, 'Bi-LSTM Layer 1\nhidden=128\ndropout=0.2', colors['lstm'])
    draw_arrow(6, 7.25, 7, 7.25)

    # LSTM 层 2 (双向)
    draw_box(7, 6, 2.5, 2.5, 'Bi-LSTM Layer 2\nhidden=128\ndropout=0.2', colors['lstm'])
    draw_arrow(9.5, 7.25, 10.5, 7.25)

    # 全连接层
    draw_box(10.5, 6.5, 2, 1.5, 'Dense\n256 → 64 → 1', colors['dense'])
    draw_arrow(12.5, 7.25, 13, 7.25)

    # 输出层
    draw_box(13, 6.5, 1, 1.5, 'Output', colors['output'])

    # 标题
    ax.text(7, 9.5, 'LSTM 天然气产量预测模型架构',
            ha='center', fontsize=16, fontweight='bold')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='输入层'),
        mpatches.Patch(facecolor=colors['lstm'], edgecolor='black', label='LSTM层'),
        mpatches.Patch(facecolor=colors['dense'], edgecolor='black', label='全连接层'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='输出层')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 3. 预测结果对比曲线图
# ==========================================
def plot_prediction_comparison(y_true, predictions_dict, title='天然气产量预测结果对比',
                                xlabel='时间步', ylabel='产量 (10⁴m³)',
                                save_name='03_prediction_comparison'):
    """
    绘制多模型预测结果对比图

    Args:
        y_true: 真实值数组
        predictions_dict: {'模型名': 预测值数组, ...}
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    colors = {'真实值': 'black', 'LSTM': 'blue', 'Informer': 'red',
              'Bi-LSTM': 'green', '优化LSTM': 'purple'}

    ax1 = axes[0]
    x = range(len(y_true))

    # 绘制真实值
    ax1.plot(x, y_true, label='真实值', color=colors['真实值'],
             linewidth=2, alpha=0.9, zorder=5)

    # 绘制各模型预测值
    for model_name, y_pred in predictions_dict.items():
        color = colors.get(model_name, 'gray')
        ax1.plot(x, y_pred, label=f'{model_name}预测',
                color=color, linewidth=1.5, alpha=0.7, linestyle='--')

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10, ncol=len(predictions_dict)+1)
    ax1.grid(True, alpha=0.3)

    # 残差图
    ax2 = axes[1]
    for model_name, y_pred in predictions_dict.items():
        residuals = y_true - y_pred
        color = colors.get(model_name, 'gray')
        ax2.plot(x, residuals, label=f'{model_name}残差',
                color=color, alpha=0.6, linewidth=1)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('残差', fontsize=12)
    ax2.set_title('预测残差分析', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 4. 残差分析图
# ==========================================
def plot_residual_analysis(y_true, y_pred, model_name='LSTM',
                            save_name='04_residual_analysis'):
    """绘制详细的残差分析图（4个子图）"""

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 残差时序图
    ax1 = axes[0, 0]
    ax1.plot(residuals, alpha=0.7, color='steelblue', linewidth=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零线')
    ax1.fill_between(range(len(residuals)), residuals, alpha=0.3, color='lightblue')
    ax1.set_title('残差时序分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('残差值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差直方图 + 正态分布拟合
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(residuals, bins=30, density=True,
                                   alpha=0.7, color='skyblue', edgecolor='black')

    # 拟合正态分布
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax2.plot(x, p, 'r-', linewidth=2, label=f'正态拟合: μ={mu:.3f}, σ={std:.3f}')

    ax2.set_title('残差分布直方图', fontsize=12, fontweight='bold')
    ax2.set_xlabel('残差值')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q图（正态性检验）
    ax3 = axes[1, 0]
    stats.probplot(residuals.flatten(), dist="norm", plot=ax3)
    ax3.set_title('Q-Q图 (正态性检验)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 预测值 vs 残差散点图
    ax4 = axes[1, 1]
    ax4.scatter(y_pred, residuals, alpha=0.5, color='green', s=20)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('预测值 vs 残差 (异方差性检验)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('预测值')
    ax4.set_ylabel('残差')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} 模型残差诊断分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 5. 评估指标对比图
# ==========================================
def plot_metrics_comparison(metrics_dict, save_name='05_metrics_comparison'):
    """
    绘制多模型评估指标对比图

    Args:
        metrics_dict: {
            'LSTM': {'MAE': 0.5, 'RMSE': 0.8, 'R2': 0.85, 'MAPE': 5.2},
            'Informer': {...}
        }
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [metrics_dict[m][metric] for m in models]

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 设置标题和标签
        metric_descriptions = {
            'MAE': '平均绝对误差 (MAE)\n(越小越好)',
            'RMSE': '均方根误差 (RMSE)\n(越小越好)',
            'R2': '决定系数 R²\n(越接近1越好)',
            'MAPE': '平均绝对百分比误差 (MAPE)\n(%)'
        }
        ax.set_title(metric_descriptions.get(metric, metric), fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 对于R2，添加参考线
        if metric == 'R2':
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.8)')
            ax.legend(fontsize=8)

    plt.suptitle('模型性能评估指标对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 6. 数据预处理流程图
# ==========================================
def plot_data_preprocessing_pipeline(save_name='06_preprocessing_pipeline'):
    """绘制数据预处理流程架构图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 颜色定义
    colors = {
        'raw': '#FFE5E5',        # 浅红
        'clean': '#FFF4E5',      # 浅橙
        'feature': '#E5F5E5',    # 浅绿
        'model': '#E5F0FF',      # 浅蓝
        'arrow': '#5D6D7E'
    }

    def draw_stage(x, y, w, h, title, items, color):
        """绘制一个处理阶段框"""
        # 外框
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.3",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)

        # 标题
        ax.text(x + w/2, y + h - 0.4, title, ha='center', va='center',
               fontsize=11, fontweight='bold', color='darkslategray')

        # 分隔线
        ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.7, y + h - 0.7],
               'k-', linewidth=1)

        # 项目列表
        for i, item in enumerate(items):
            ax.text(x + w/2, y + h - 1.2 - i*0.5, f"• {item}",
                   ha='center', va='center', fontsize=9)

    def draw_arrow(x1, y1, x2, y2, label=None):
        """绘制箭头"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2.5))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                   fontsize=9, style='italic', color='dimgray')

    # 绘制各个阶段
    # 1. 原始数据
    draw_stage(0.5, 6.5, 3, 2.5, '① 原始数据采集',
              ['油田Excel数据', '井口压力、产量', '时间序列对齐'], colors['raw'])
    draw_arrow(3.5, 7.75, 4.5, 7.75, '数据加载')

    # 2. 数据清洗
    draw_stage(4.5, 6.5, 3, 2.5, '② 数据预处理',
              ['3-Sigma异常值剔除', '缺失值线性插值', 'StandardScaler归一化'], colors['clean'])
    draw_arrow(7.5, 7.75, 8.5, 7.75, '特征构建')

    # 3. 特征工程
    draw_stage(8.5, 6.5, 3, 2.5, '③ 特征工程',
              ['滞后特征(lag=1,3,7)', '滑动窗口统计(3,7)', '时序窗口构建(seq=30)'], colors['feature'])
    draw_arrow(11.5, 7.75, 12.5, 7.75, '模型输入')

    # 4. 模型预测（左右布局）
    # LSTM
    draw_stage(1.5, 2.5, 4, 2.5, '④-a LSTM预测模型',
              ['双向LSTM(128×2层)', 'Attention机制', 'Dropout=0.2', 'MSE损失函数'], colors['model'])

    # Informer
    draw_stage(9.5, 2.5, 4, 2.5, '④-b Informer预测模型',
              ['ProbSparse Attention', '自注意力蒸馏', '生成式解码器', '长时序预测'], colors['model'])

    # 合并输出
    draw_arrow(5.5, 3.75, 6.5, 3.75)
    draw_arrow(9.5, 3.75, 8.5, 3.75)

    # 输出
    draw_stage(6.5, 2.5, 2, 2.5, '⑤ 预测输出',
              ['产量预测值', '误差评估', '可视化'], '#F5EEF8')

    # 整体标题
    ax.text(8, 9.5, '天然气产量预测系统 - 整体技术路线图',
           ha='center', fontsize=18, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 7. 训练过程曲线图
# ==========================================
def plot_training_curves(history, save_name='07_training_curves'):
    """
    绘制训练过程曲线

    Args:
        history: dict with 'train_loss', 'val_loss', 'train_mae', 'val_mae' etc.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='训练损失')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='验证损失')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('训练/验证损失曲线', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MAE曲线
    ax2 = axes[1]
    if 'train_mae' in history:
        ax2.plot(epochs, history['train_mae'], 'g-', linewidth=2, label='训练MAE')
    if 'val_mae' in history:
        ax2.plot(epochs, history['val_mae'], 'orange', linewidth=2, label='验证MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('训练/验证MAE曲线', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 8. 综合评估仪表盘
# ==========================================
def create_evaluation_dashboard(y_true, y_pred, model_name='LSTM',
                                 metrics=None, save_name='08_dashboard'):
    """
    创建综合评估仪表盘（6个子图）
    """
    residuals = y_true - y_pred

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 预测 vs 真实值散点图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors='none', s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title('预测准确性散点图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差直方图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('残差')
    ax2.set_ylabel('频数')
    ax2.set_title('残差分布直方图')
    ax2.grid(True, alpha=0.3)

    # 3. 指标展示（文本）
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    if metrics:
        metrics_text = f"""
        模型评估指标
        =================

        MAE:  {metrics.get('MAE', 'N/A'):.4f}
        RMSE: {metrics.get('RMSE', 'N/A'):.4f}
        R²:   {metrics.get('R2', 'N/A'):.4f}
        MAPE: {metrics.get('MAPE', 'N/A'):.2f}%

        =================
        模型: {model_name}
        样本数: {len(y_true)}
        """
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='center',
                fontfamily=CHINESE_FONT or 'sans-serif',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 时间序列预测曲线（取前200个点展示）
    ax4 = fig.add_subplot(gs[1, :2])
    n_show = min(200, len(y_true))
    x = range(n_show)
    ax4.plot(x, y_true[:n_show], label='真实值', color='black', linewidth=1.5)
    ax4.plot(x, y_pred[:n_show], label=f'{model_name}预测',
            color='red', linewidth=1.5, alpha=0.8, linestyle='--')
    ax4.fill_between(x, y_true[:n_show], y_pred[:n_show],
                     alpha=0.2, color='gray', label='误差区间')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('产量')
    ax4.set_title('时序预测结果对比（局部）')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # 5. 残差 vs 拟合值散点图
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(y_pred, residuals, alpha=0.5, edgecolors='none', s=20, color='purple')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('拟合值 (预测值)')
    ax5.set_ylabel('残差')
    ax5.set_title('残差 vs 拟合值散点图')
    ax5.grid(True, alpha=0.3)

    # 6. 误差分布箱线图
    ax6 = fig.add_subplot(gs[2, :])
    abs_errors = np.abs(residuals)
    error_data = [abs_errors[abs_errors < np.percentile(abs_errors, 25)],
                  abs_errors[(abs_errors >= np.percentile(abs_errors, 25)) &
                            (abs_errors < np.percentile(abs_errors, 50))],
                  abs_errors[(abs_errors >= np.percentile(abs_errors, 50)) &
                            (abs_errors < np.percentile(abs_errors, 75))],
                  abs_errors[abs_errors >= np.percentile(abs_errors, 75)]]

    bp = ax6.boxplot([abs_errors], vert=False, patch_artist=True,
                     labels=['绝对误差'],
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

    # 添加统计信息文本
    stats_text = f"""
    误差统计信息:
    均值: {np.mean(abs_errors):.4f}
    中位数: {np.median(abs_errors):.4f}
    标准差: {np.std(abs_errors):.4f}
    最大值: {np.max(abs_errors):.4f}
    95%分位数: {np.percentile(abs_errors, 95):.4f}
    """
    ax6.text(0.98, 0.5, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='right',
            fontfamily=CHINESE_FONT or 'sans-serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax6.set_xlabel('绝对误差值', fontsize=11)
    ax6.set_title('预测误差分布箱线图', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'{model_name} 模型综合评估仪表盘', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 9. 消融实验对比图
# ==========================================
def plot_ablation_study(results, save_name='09_ablation_study'):
    """
    绘制消融实验结果对比

    Args:
        results: {
            '完整模型': {'RMSE': 0.5, 'MAE': 0.4, 'R2': 0.9},
            '去掉滞后特征': {...},
            '去掉滑动统计': {...},
            '单向LSTM': {...}
        }
    """
    models = list(results.keys())
    metrics = list(results[models[0]].keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[m][metric] for m in models]

        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 标注最优值
            if metric in ['R2']:
                if val == max(values):
                    ax.annotate('最优', xy=(i, val), xytext=(i, val + 0.02),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               fontsize=9, color='red', fontweight='bold')
            else:
                if val == min(values):
                    ax.annotate('最优', xy=(i, val), xytext=(i, val + 0.02),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               fontsize=9, color='red', fontweight='bold')

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('消融实验结果对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close()
    return fig


# ==========================================
# 主函数：生成所有图表
# ==========================================
def generate_all_thesis_plots():
    """生成论文所需的所有图表"""
    print("="*60)
    print("开始生成毕业论文图表...")
    print("="*60)

    # 1. LSTM 架构图
    print("[1/9] 生成 LSTM 架构图...")
    plot_lstm_architecture()

    # 2. 数据预处理流程图
    print("[2/9] 生成数据预处理流程图...")
    plot_data_preprocessing_pipeline()

    # 以下是使用示例数据生成其他图表的示例
    # 实际使用时，请传入你的真实数据

    print("\n" + "="*60)
    print("基础架构图生成完成！")
    print("="*60)
    print(f"\n图表保存位置: {OUTPUT_DIR}")
    print("\n其他图表需要真实数据，请使用以下示例代码：")
    print("""
    # 特征相关性热力图
    plot_feature_correlation(df, ['wellhead_press', 'gas_volume', 'lag_1', 'roll_mean_7'])

    # 预测结果对比图
    plot_prediction_comparison(y_true, {'LSTM': y_lstm, 'Informer': y_informer})

    # 残差分析图
    plot_residual_analysis(y_true, y_pred, model_name='LSTM')

    # 评估指标对比图
    plot_metrics_comparison({
        'LSTM': {'MAE': 0.5, 'RMSE': 0.8, 'R2': 0.85, 'MAPE': 5.2},
        'Informer': {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90, 'MAPE': 4.1}
    })

    # 消融实验对比图
    plot_ablation_study({
        '完整模型': {'RMSE': 0.5, 'MAE': 0.4, 'R2': 0.9},
        '去掉滞后特征': {'RMSE': 0.7, 'MAE': 0.55, 'R2': 0.82},
        '去掉滑动统计': {'RMSE': 0.65, 'MAE': 0.5, 'R2': 0.84},
        '单向LSTM': {'RMSE': 0.6, 'MAE': 0.48, 'R2': 0.86}
    })
    """)


if __name__ == "__main__":
    generate_all_thesis_plots()
