import numpy as np
import pandas as pd

class DiagnosisRules:
    """
    积液诊断专家规则类：基于物理判据对气井状态进行分类。
    分类结果：0: 正常, 1: 带水生产, 2: 积液, 3: 关井, -1: 错误数据
    """
    
    @staticmethod
    def calculate_critical_flow(wellhead_press):
        """
        基于油压计算临界产气量 (Turner 模型简化版)。
        """
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0.56, 0.80, 0.98, 1.14, 1.28, 1.41, 1.53, 1.64, 1.75, 1.85]
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        return p(wellhead_press)

    @classmethod
    def diagnose_single_row(cls, row):
        """
        对单行生产数据进行规则判定。
        """
        gas = row.get('gas_volume', np.nan)
        whp = row.get('wellhead_press', np.nan)
        # 假设我们有套压数据，如果没有则默认压差正常
        csg_p = row.get('casing_press', whp + 1.0) 
        prod_time = row.get('prod_hours', 24)

        if pd.isna(gas) or pd.isna(whp):
            return -1 # 数据缺失
        
        if prod_time < 12:
            return 3 # 关井状态

        diff = csg_p - whp
        critical_gas = cls.calculate_critical_flow(whp)

        # 核心逻辑
        if whp < 2.5:
            if diff <= 2.0:
                return 0 # 正常
            else:
                return 2 if (critical_gas - gas) > 0.1 else 1 # 积液 vs 带水
        else:
            if diff <= 3.5:
                return 0 # 正常
            else:
                return 2 if (critical_gas - gas) > 0.1 else 1 # 积液 vs 带水

    @classmethod
    def apply_rules(cls, df):
        """
        批量应用规则并生成标签。
        """
        df_res = df.copy()
        df_res['diag_label'] = df_res.apply(cls.diagnose_single_row, axis=1)
        
        # 标签映射说明
        label_map = {0: 'Normal', 1: 'WorkWithWater', 2: 'Accumulation', 3: 'Closed', -1: 'Error'}
        df_res['diag_state'] = df_res['diag_label'].map(label_map)
        
        return df_res
