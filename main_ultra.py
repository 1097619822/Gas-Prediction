import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config
from evaluation.metrics import Evaluator
from models.lstm_model import LSTMPredictor, LSTMTrainer
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.data_loader import DataLoader as DL
from pre_processing.feature_eng import FeatureEngineer as FE
from visualization.plot_results import Visualizer


def main():
    print('=' * 70)
    print('毕业设计实验: Optimized Bi-LSTM production prediction')
    print('=' * 70)

    well_files = DL.get_all_well_files()
    if not well_files:
        raise RuntimeError(f'No Excel files found in {config.DATA_DIR}')

    target_well = well_files[0]
    print(f'[Data] Processing target well: {target_well}')

    df = DL.load_well_data(target_well)
    df = DC.unify_formats(df)
    df = DC.remove_outliers_3sigma(df, ['wellhead_press', 'gas_volume'])
    df = DC.handle_missing_values(df, ['wellhead_press', 'gas_volume'])

    df = FE.add_lagged_features(df, 'gas_volume')
    df = FE.add_rolling_features(df, ['wellhead_press'])
    df = df.dropna().copy()

    features = ['wellhead_press', 'gas_volume_lag_1', 'wellhead_press_roll_7_mean']
    target = 'gas_volume'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df[features].values)
    y_scaled = scaler_y.fit_transform(df[target].values.reshape(-1, 1))

    seq_len = 30
    model = LSTMPredictor(input_size=len(features), hidden_size=128, num_layers=2, bidirectional=True)
    trainer = LSTMTrainer(model, lr=0.001)
    X_tensor, y_tensor = trainer.prepare_data(X_scaled, y_scaled, seq_len=seq_len)

    split = int(len(X_tensor) * 0.8)
    X_train, y_train = X_tensor[:split], y_tensor[:split]
    X_test, y_test = X_tensor[split:], y_tensor[split:]

    print(f'[Data] Train samples: {len(X_train)}, test samples: {len(X_test)}')
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    trainer.train(train_loader, epochs=50)

    model_path = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2.pth')
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2_checkpoint.pth')
    scaler_x_path = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2_scaler_x.pkl')
    scaler_y_path = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2_scaler_y.pkl')
    trainer.save_model(model_path)
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    y_pred_scaled = trainer.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test.numpy())

    metrics = Evaluator.calculate_all_metrics(y_true, y_pred)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'features': features,
        'target': target,
        'seq_len': seq_len,
        'model_config': {
            'input_size': len(features),
            'hidden_size': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.2,
        },
        'scaler_x_path': scaler_x_path,
        'scaler_y_path': scaler_y_path,
        'metrics': metrics,
        'target_well': target_well,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'[LSTM] Checkpoint saved to: {checkpoint_path}')
    print(f'[LSTM] Scalers saved to: {scaler_x_path}, {scaler_y_path}')
    Evaluator.print_results(metrics, model_name='Optimized Bi-LSTM (Ultra)')

    print('\n[Visualization] Exporting final comparison curve...')
    res_df = pd.DataFrame({
        'index': range(len(y_true)),
        'actual': y_true.flatten(),
        'predicted': y_pred.flatten(),
    })
    Visualizer.plot_time_series(
        res_df,
        'index',
        ['actual', 'predicted'],
        title=f'{target_well} production prediction comparison',
    )

    print('\n' + '=' * 70)
    print('Experiment completed. Informer remains a separate comparison path in this project.')
    print('=' * 70)
    return metrics


if __name__ == '__main__':
    main()

