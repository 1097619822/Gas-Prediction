import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import config
from pre_processing.data_loader import DataLoader as DL
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.diagnosis_rules import DiagnosisRules as DR
from models.classifier import LSTMClassifier, ClassificationTrainer

def prepare_sequences(X, y, seq_len=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long).view(-1)
    return X_tensor, y_tensor

def main():
    print("="*70)
    print("馃殌 姣曚笟璁捐缁堟瀬瀹為獙锛氬叏閲忎簳绉恫璇婃柇妯″瀷璁粌")
    print("="*70)

    well_files = DL.get_all_well_files()
    all_x, all_y = [], []
    
    # 1. 鍏ㄩ噺鎵弿骞舵瀯寤烘贩鍚堟暟鎹泦
    print(f"[Step 1] 姝ｅ湪鎵弿鍏ㄩ噺浜?({len(well_files)} 鍙? 浠ユ彁鍙栫Н娑茬壒寰?..")
    
    for filename in well_files:
        try:
            df = DL.load_well_data(filename)
            df_labeled = DR.apply_rules(df)
            
            # 杩囨护鎺夌己澶卞€?
            df_valid = df_labeled[df_labeled['diag_label'] >= 0].copy()
            if len(df_valid) < 20: continue
            
            # 鐗瑰緛宸ョ▼ (鏈€绠€鐗瑰緛鐢ㄤ簬璇婃柇)
            features = ['wellhead_press', 'gas_volume']
            X_raw = df_valid[features].values
            
            # 灞€閮ㄥ綊涓€鍖?(闃叉孩鍑?
            X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
            X_norm = (X_raw - X_min) / (X_max - X_min + 1e-6)
            y_raw = df_valid['diag_label'].values
            
            # Build sequence samples
            tx, ty = prepare_sequences(X_norm, y_raw, seq_len=7)
            
            if len(tx) > 0:
                all_x.append(tx)
                all_y.append(ty)
                # print(f" - {filename}: 鎴愬姛 ({len(tx)} 鏍锋湰)")
        except Exception as exc:
            print(f'[Warn] Skip {filename}: {exc}')
            continue

    if not all_x:
        print("\n[Fatal] 鏁版嵁姹囨€诲け璐ワ紒")
        return

    X_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)
    
    print(f"\n[鏁版嵁闆嗙粺璁:")
    print(f" - 鎬绘牱鏈暟: {len(X_total)}")
    unique, counts = np.unique(y_total.numpy(), return_counts=True)
    state_map = {0: '姝ｅ父', 1: '甯︽按', 2: '绉恫', 3: '鍏充簳'}
    for u, c in zip(unique, counts):
        print(f" - {state_map[int(u)]:4s}: {c} 鏍锋湰")

    # 2. 鍚姩鍒嗙被璁粌 (Bi-LSTM)
    print(f"\n[Step 2] 姝ｅ湪鍚姩娣卞害瀛︿範鍒嗙被妯″瀷璁粌...")
    model = LSTMClassifier(input_size=2, num_classes=4)
    trainer = ClassificationTrainer(model, lr=0.002)
    epochs = int(os.getenv('DIAG_EPOCHS', '30'))
    
    train_loader = DataLoader(TensorDataset(X_total, y_total), batch_size=64, shuffle=True)
    trainer.train_step(train_loader, None, epochs=epochs)

    # 3. 淇濆瓨鏈€缁堟垚鏋?
    save_path = os.path.join(config.OUTPUT_DIR, "final_diagnosis_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[Success] 绉恫璇婃柇妯″瀷宸蹭繚瀛樿嚦: {save_path}")

    print("\n" + "="*70)
    print("馃帀 鎭枩锛佷綘鐨勬瘯涓氳璁℃墍鏈夋牳蹇冪畻娉曟ā鍧楀凡鍏ㄩ儴鎵撻€氾紒")
    print("="*70)

if __name__ == "__main__":
    main()




