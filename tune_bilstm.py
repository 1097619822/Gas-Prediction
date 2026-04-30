"""
Bi-LSTM hyperparameter tuning for the production prediction task.

Run from src_v2:
    python -u tune_bilstm.py

Useful environment variables:
    TUNE_EPOCHS=35
    TUNE_FINAL_EPOCHS=80
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
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

FEATURES = [
    'wellhead_press',
    'casing_press',
    'prod_hours',
    'pressure_diff',
    'gas_volume_lag_1',
    'gas_volume_lag_3',
    'gas_volume_lag_7',
    'wellhead_press_roll_3_mean',
    'wellhead_press_roll_7_mean',
]
TARGET = 'gas_volume'
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataframe(well_file):
    df = DL.load_well_data(well_file)
    df = DC.unify_formats(df)
    df = DC.remove_outliers_3sigma(df, ['wellhead_press', 'casing_press', 'gas_volume', 'prod_hours'])
    df = DC.handle_missing_values(df, ['wellhead_press', 'casing_press', 'gas_volume', 'prod_hours'])
    df['pressure_diff'] = df['casing_press'] - df['wellhead_press']
    df = FE.add_lagged_features(df, TARGET)
    df = FE.add_rolling_features(df, ['wellhead_press'])
    df = df.dropna().copy()
    available = [feature for feature in FEATURES if feature in df.columns]
    return df, available


def build_sequences(df, features, seq_len):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df[features].values)
    y_scaled = scaler_y.fit_transform(df[TARGET].values.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y_scaled[i + seq_len])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    return {
        'X_train': X_seq[:train_end],
        'y_train': y_seq[:train_end],
        'X_val': X_seq[train_end:val_end],
        'y_val': y_seq[train_end:val_end],
        'X_test': X_seq[val_end:],
        'y_test': y_seq[val_end:],
        'X_trainval': X_seq[:val_end],
        'y_trainval': y_seq[:val_end],
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
    }


def train_model(data, input_size, hidden_size, num_layers, lr, dropout, epochs):
    set_seed()
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        dropout=dropout,
    )
    trainer = LSTMTrainer(model, lr=lr)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(data['X_train'], dtype=torch.float32),
            torch.tensor(data['y_train'], dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    trainer.train(train_loader, epochs=epochs)
    model.eval()
    with torch.no_grad():
        val_pred_scaled = model(torch.tensor(data['X_val'], dtype=torch.float32)).numpy()
    val_pred = data['scaler_y'].inverse_transform(val_pred_scaled).flatten()
    val_true = data['scaler_y'].inverse_transform(data['y_val']).flatten()
    metrics = Evaluator.calculate_all_metrics(val_true, val_pred)
    return model, metrics


def train_final_model(data, input_size, config_item, epochs):
    set_seed()
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=config_item['hidden_size'],
        num_layers=config_item['num_layers'],
        bidirectional=True,
        dropout=config_item['dropout'],
    )
    trainer = LSTMTrainer(model, lr=config_item['lr'])
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(data['X_trainval'], dtype=torch.float32),
            torch.tensor(data['y_trainval'], dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    trainer.train(train_loader, epochs=epochs)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(torch.tensor(data['X_test'], dtype=torch.float32)).numpy()
    y_pred = data['scaler_y'].inverse_transform(test_pred_scaled).flatten()
    y_true = data['scaler_y'].inverse_transform(data['y_test']).flatten()
    metrics = Evaluator.calculate_all_metrics(y_true, y_pred)
    mask = np.abs(y_true) > 0.1
    metrics['MAPE_filtered'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return model, y_true, y_pred, metrics


def main():
    set_seed()
    well_files = DL.get_all_well_files()
    if not well_files:
        raise RuntimeError(f'No Excel files found in {config.DATA_DIR}')
    well_file = os.getenv('TUNE_WELL', well_files[0])
    tune_epochs = int(os.getenv('TUNE_EPOCHS', '25'))
    final_epochs = int(os.getenv('TUNE_FINAL_EPOCHS', '80'))

    print('=' * 70)
    print('Bi-LSTM hyperparameter tuning')
    print('=' * 70)
    print(f'[Data] Well: {well_file}')
    print(f'[Train] tune_epochs={tune_epochs}, final_epochs={final_epochs}')

    df, features = prepare_dataframe(well_file)
    print(f'[Data] rows={len(df)}, features={features}')

    candidates = [
        {'seq_len': 7, 'hidden_size': 64, 'num_layers': 1, 'lr': 0.001, 'dropout': 0.0},
        {'seq_len': 7, 'hidden_size': 128, 'num_layers': 1, 'lr': 0.001, 'dropout': 0.0},
        {'seq_len': 14, 'hidden_size': 64, 'num_layers': 1, 'lr': 0.001, 'dropout': 0.0},
        {'seq_len': 14, 'hidden_size': 128, 'num_layers': 2, 'lr': 0.001, 'dropout': 0.2},
        {'seq_len': 30, 'hidden_size': 128, 'num_layers': 2, 'lr': 0.001, 'dropout': 0.2},
        {'seq_len': 14, 'hidden_size': 128, 'num_layers': 2, 'lr': 0.0005, 'dropout': 0.2},
    ]

    results = []
    best = None
    best_data = None
    for idx, candidate in enumerate(candidates, start=1):
        print('\n' + '-' * 70)
        print(f'[Candidate {idx}/{len(candidates)}] {candidate}')
        data = build_sequences(df, features, candidate['seq_len'])
        _, val_metrics = train_model(
            data,
            input_size=len(features),
            hidden_size=candidate['hidden_size'],
            num_layers=candidate['num_layers'],
            lr=candidate['lr'],
            dropout=candidate['dropout'],
            epochs=tune_epochs,
        )
        row = dict(candidate)
        row.update({f'val_{k}': v for k, v in val_metrics.items()})
        results.append(row)
        print(f"[Val] RMSE={val_metrics['RMSE']:.4f}, MAE={val_metrics['MAE']:.4f}, R2={val_metrics['R2']:.4f}")
        if best is None or val_metrics['RMSE'] < best['val_RMSE']:
            best = row
            best_data = data

    print('\n' + '=' * 70)
    print(f'[Best] {best}')
    print('=' * 70)

    best_config = {
        'seq_len': int(best['seq_len']),
        'hidden_size': int(best['hidden_size']),
        'num_layers': int(best['num_layers']),
        'lr': float(best['lr']),
        'dropout': float(best['dropout']),
    }
    best_data = build_sequences(df, features, best_config['seq_len'])
    final_model, y_true, y_pred, test_metrics = train_final_model(best_data, len(features), best_config, final_epochs)
    print(f"[Test] RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}, R2={test_metrics['R2']:.4f}, MAPE_filtered={test_metrics['MAPE_filtered']:.2f}%")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuning_results.xlsx')
    pred_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuned_predictions.xlsx')
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuned_checkpoint.pth')
    weight_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuned_model.pth')
    scaler_x_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuned_scaler_x.pkl')
    scaler_y_path = os.path.join(config.OUTPUT_DIR, 'bilstm_tuned_scaler_y.pkl')

    pd.DataFrame(results).to_excel(result_path, index=False)
    pd.DataFrame({'actual': y_true, 'Bi-LSTM-Tuned': y_pred}).to_excel(pred_path, index=False)
    torch.save(final_model.state_dict(), weight_path)
    joblib.dump(best_data['scaler_x'], scaler_x_path)
    joblib.dump(best_data['scaler_y'], scaler_y_path)
    torch.save({
        'well_file': well_file,
        'features': features,
        'target': TARGET,
        'seq_len': best_config['seq_len'],
        'model_config': {
            'input_size': len(features),
            'hidden_size': best_config['hidden_size'],
            'num_layers': best_config['num_layers'],
            'bidirectional': True,
            'dropout': best_config['dropout'],
        },
        'lr': best_config['lr'],
        'model_state_dict': final_model.state_dict(),
        'scaler_x_path': scaler_x_path,
        'scaler_y_path': scaler_y_path,
        'metrics': test_metrics,
        'tuning_results_path': result_path,
        'predictions_path': pred_path,
    }, checkpoint_path)

    print('\n[Saved]')
    for path in [result_path, pred_path, weight_path, scaler_x_path, scaler_y_path, checkpoint_path]:
        print(f'  {path}')


if __name__ == '__main__':
    main()
