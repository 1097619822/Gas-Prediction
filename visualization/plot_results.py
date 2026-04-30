import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
import numpy as np

# 学术风格配置
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


sns.set_theme(style="whitegrid")
CHINESE_FONT = setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """
    绘图逻辑封装：支持单曲线和多曲线对比。
    """
    @staticmethod
    def plot_time_series(df, date_col, target_cols, title="产量趋势图"):
        """
        :param target_cols: 可以是字符串(单列)或列表(多列)
        """
        plt.figure(figsize=(12, 6))
        
        if isinstance(target_cols, list):
            # 如果是多列，将其转换为长格式以便 Seaborn 绘制多线
            df_melted = df.melt(id_vars=[date_col], value_vars=target_cols, 
                                var_name='Type', value_name='Value')
            sns.lineplot(data=df_melted, x=date_col, y='Value', hue='Type', marker='o')
        else:
            sns.lineplot(data=df, x=date_col, y=target_cols, marker='o')
            
        plt.title(title, fontsize=15)
        plt.xlabel("日期/样本序号")
        plt.ylabel("产气量")
        plt.legend(title="对比项")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df, columns, title="特征相关性分析"):
        corr = df[columns].corr()
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.show()
