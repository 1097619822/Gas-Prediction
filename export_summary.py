import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing.data_loader import DataLoader as DL
from pre_processing.diagnosis_rules import DiagnosisRules as DR

def export_all_well_summaries():
    """
    全自动扫描 31 口井，生成积液诊断统计汇总表。
    """
    well_files = DL.get_all_well_files()
    summary_list = []
    
    print(f"正在汇总全量井数据，共 {len(well_files)} 口...")
    
    for filename in well_files:
        try:
            df = DL.load_well_data(filename)
            df_labeled = DR.apply_rules(df)
            
            stats = df_labeled['diag_state'].value_counts()
            
            summary_list.append({
                '井号': filename,
                '总记录数': len(df),
                '正常生产天数': stats.get('Normal', 0),
                '积液天数': stats.get('Accumulation', 0),
                '平均产气量': df['gas_volume'].mean(),
                '平均油压': df['wellhead_press'].mean()
            })
        except:
            continue
            
    summary_df = pd.DataFrame(summary_list)
    output_path = r"D:\毕业设计\Project\src_v2\processed_results\全井生产汇总分析表.xlsx"
    summary_df.to_excel(output_path, index=False)
    print(f"🎉 汇总报表已成功导出至: {output_path}")

if __name__ == "__main__":
    export_all_well_summaries()
