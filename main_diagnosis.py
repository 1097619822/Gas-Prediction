import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from pre_processing.data_loader import DataLoader as DL
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.diagnosis_rules import DiagnosisRules as DR
from models.classifier import LSTMClassifier, ClassificationTrainer
from models.lstm_model import LSTMTrainer

def main():
    print("="*70)
    print("🚀 毕业设计终极实验：全量井积液诊断模型训练")
    print("="*70)

    well_files = DL.get_all_well_files()
    all_x, all_y = [], []
    
    # 1. 全量扫描并构建混合数据集
    print(f"[Step 1] 正在扫描全量井 ({len(well_files)} 口) 以提取积液特征...")
    
    for filename in well_files:
        try:
            df = DL.load_well_data(filename)
            df_labeled = DR.apply_rules(df)
            
            # 过滤掉缺失值
            df_valid = df_labeled[df_labeled['diag_label'] >= 0].copy()
            if len(df_valid) < 20: continue
            
            # 特征工程 (最简特征用于诊断)
            features = ['wellhead_press', 'gas_volume']
            X_raw = df_valid[features].values
            
            # 局部归一化 (防溢出)
            X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
            X_norm = (X_raw - X_min) / (X_max - X_min + 1e-6)
            y_raw = df_valid['diag_label'].values
            
            # 时序切分
            temp_trainer = LSTMTrainer(None)
            tx, ty = temp_trainer.prepare_data(X_norm, y_raw, seq_len=7)
            
            if len(tx) > 0:
                all_x.append(tx)
                all_y.append(ty)
                # print(f" - {filename}: 成功 ({len(tx)} 样本)")
        except:
            continue

    if not all_x:
        print("\n[Fatal] 数据汇总失败！")
        return

    X_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)
    
    print(f"\n[数据集统计]:")
    print(f" - 总样本数: {len(X_total)}")
    unique, counts = np.unique(y_total.numpy(), return_counts=True)
    state_map = {0: '正常', 1: '带水', 2: '积液', 3: '关井'}
    for u, c in zip(unique, counts):
        print(f" - {state_map[int(u)]:4s}: {c} 样本")

    # 2. 启动分类训练 (Bi-LSTM)
    print(f"\n[Step 2] 正在启动深度学习分类模型训练...")
    model = LSTMClassifier(input_size=2, num_classes=4)
    trainer = ClassificationTrainer(model, lr=0.002)
    
    train_loader = DataLoader(TensorDataset(X_total, y_total), batch_size=64, shuffle=True)
    trainer.train_step(train_loader, None, epochs=30)

    # 3. 保存最终成果
    save_path = os.path.join(config.OUTPUT_DIR, "final_diagnosis_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[Success] 积液诊断模型已保存至: {save_path}")

    print("\n" + "="*70)
    print("🎉 恭喜！你的毕业设计所有核心算法模块已全部打通！")
    print("="*70)

if __name__ == "__main__":
    main()
