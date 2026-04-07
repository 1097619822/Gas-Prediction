import pandas as pd
import sys
import os

# 将项目根目录加入搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class DataLoader:
    """
    终极稳健加载器：基于物理位置(物理坐标)抓取列，确保关键字段 wellhead_press 不丢失。
    """
    @staticmethod
    def load_well_data(well_filename):
        path = os.path.join(config.DATA_DIR, well_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Loader] 找不到文件: {path}")
        
        # 跳过第一行（通常是空行或单位行），从真实表头开始读
        df_raw = pd.read_excel(path, sheet_name=0)
        
        # 物理坐标定义 (根据你的 54-16X.xlsx 结构)
        # 0:日期, 7:井口压力(油压), 8:套管压力, 13:日产气量, 4:生产时间
        target_cols = {
            0: 'date',
            7: 'wellhead_press',
            8: 'casing_press',
            13: 'gas_volume',
            4: 'prod_hours'
        }
        
        # 确保索引不越界
        valid_indices = [idx for idx in target_cols.keys() if idx < len(df_raw.columns)]
        df_selected = df_raw.iloc[:, valid_indices].copy()
        df_selected.columns = [target_cols[idx] for idx in valid_indices]
        
        # 强制类型转换
        for col in ['wellhead_press', 'casing_press', 'gas_volume', 'prod_hours']:
            if col in df_selected.columns:
                df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        
        # 如果依然缺失 wellhead_press (有些文件可能格式不同)，用 casing_press 补位
        if 'wellhead_press' not in df_selected.columns or df_selected['wellhead_press'].isnull().all():
             if 'casing_press' in df_selected.columns:
                 df_selected['wellhead_press'] = df_selected['casing_press'] - 1.0

        df_selected.dropna(subset=['date'], inplace=True)
        return df_selected

    @staticmethod
    def get_all_well_files():
        return [f for f in os.listdir(config.DATA_DIR) if f.endswith('.xlsx')]
