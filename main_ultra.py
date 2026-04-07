import os
import torch
import pandas as pd
import numpy as np
import config
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 导入 src_v2 核心模块
from pre_processing.data_loader import DataLoader as DL
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.feature_eng import FeatureEngineer as FE
from models.lstm_model import LSTMPredictor, LSTMTrainer
from evaluation.metrics import Evaluator
from visualization.plot_results import Visualizer

# 注意：Informer 模型由于其内部依赖较为复杂，通常作为独立进程运行或通过 Exp 类进行实例化
# 在本脚本中，我们主要对比优化后的 LSTM 与之前基准模型的提升，并为 Informer 预留接口

def main():
    print("="*70)
    print("🏆 毕业设计终极实验：优化 LSTM vs Informer (创新点展示)")
    print("="*70)

    # 1. 极严苛的数据预处理
    well_files = DL.get_all_well_files()
    target_well = well_files[0]
    print(f"[Data] 正在处理目标井: {target_well}")
    
    df = DL.load_well_data(target_well)
    df = DC.unify_formats(df)
    df = DC.remove_outliers_3sigma(df, ['wellhead_press', 'gas_volume'])
    df = DC.handle_missing_values(df, ['wellhead_press', 'gas_volume'])
    
    # 特征工程增强
    df = FE.add_lagged_features(df, 'gas_volume')
    df = FE.add_rolling_features(df, ['wellhead_press'])
    df = df.dropna().copy()
    
    features = ['wellhead_press', 'gas_volume_lag_1', 'wellhead_press_roll_7_mean']
    target = 'gas_volume'
    
    # 2. 深度归一化 (StandardScaler)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df[features].values)
    y_scaled = scaler_y.fit_transform(df[target].values.reshape(-1, 1))
    
    # 3. 准备时序窗口 (Seq_Len = 30, 学术标准长时序)
    seq_len = 30
    
    # 使用优化后的 LSTM 训练器
    model = LSTMPredictor(input_size=len(features), hidden_size=128, num_layers=2, bidirectional=True)
    trainer = LSTMTrainer(model, lr=0.001)
    
    X_tensor, y_tensor = trainer.prepare_data(X_scaled, y_scaled, seq_len=seq_len)
    
    # 划分 80/20
    split = int(len(X_tensor) * 0.8)
    X_train, y_train = X_tensor[:split], y_tensor[:split]
    X_test, y_test = X_tensor[split:], y_tensor[split:]
    
    # 4. 训练优化后的 LSTM (Bi-LSTM + Deep layers)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    trainer.train(train_loader, epochs=50) # 充分训练
    
    # 保存该版本的最优模型
    trainer.save_model(os.path.join(config.OUTPUT_DIR, "ultra_lstm_v2.pth"))

    # 5. 执行预测并反归一化
    y_pred_scaled = trainer.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test.numpy())

    # 6. 评估结果
    metrics = Evaluator.calculate_all_metrics(y_true, y_pred)
    Evaluator.print_results(metrics, model_name="Optimized Bi-LSTM (Ultra)")

    # 7. 可视化
    print("\n[Visualization] 正在导出最终对比曲线...")
    res_df = pd.DataFrame({
        '序号': range(len(y_true)),
        '真实值': y_true.flatten(),
        '优化后LSTM预测值': y_pred.flatten()
    })
    Visualizer.plot_time_series(res_df, '序号', ['真实值', '优化后LSTM预测值'], title=f"{target_well} 终极产量预测对比")

    print("\n" + "="*70)
    print("✨ 实验完成！Informer 模型的详细对比建议在论文中通过 script/ 下的 .sh 脚本运行产出。")
    print("="*70)

if __name__ == "__main__":
    main()
