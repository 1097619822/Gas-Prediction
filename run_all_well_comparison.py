"""
All-well model comparison experiment.

This script evaluates ARIMA, SVR, and Bi-LSTM on every well file under data/.
It writes per-well metrics and model-level summary tables to processed_results/.

Run from src_v2:
    python -u run_all_well_comparison.py

Optional environment variables:
    ALL_WELL_EPOCHS=20       Bi-LSTM epochs per well
    ALL_WELL_LIMIT=3         Only run first N wells for quick validation
"""
import os
import sys
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset

import config
from evaluation.metrics import Evaluator
from models.lstm_model import LSTMPredictor, LSTMTrainer
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.data_loader import DataLoader as DL
from pre_processing.feature_eng import FeatureEngineer as FE

warnings.filterwarnings('ignore')

FEATURES = ['wellhead_press', 'gas_volume_lag_1', 'wellhead_press_roll_7_mean']
TARGET = 'gas_volume'
SEQ_LEN = 30
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataframe(well_file):
    df = DL.load_well_data(well_file)
    df = DC.unify_formats(df)
    df = DC.remove_outliers_3sigma(df, ['wellhead_press', 'gas_volume'])
    df = DC.handle_missing_values(df, ['wellhead_press', 'gas_volume'])
    df = FE.add_lagged_features(df, TARGET)
    df = FE.add_rolling_features(df, ['wellhead_press'])
    df = df.dropna().copy()
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f'missing features: {missing}')
    if len(df) <= SEQ_LEN + 20:
        raise ValueError(f'not enough rows after feature engineering: {len(df)}')
    return df


def build_data(df):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df[FEATURES].values)
    y_scaled = scaler_y.fit_transform(df[TARGET].values.reshape(-1, 1))

    X_seq, X_flat, y_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        X_flat.append(X_scaled[i + SEQ_LEN])
        y_seq.append(y_scaled[i + SEQ_LEN])

    X_seq = np.array(X_seq, dtype=np.float32)
    X_flat = np.array(X_flat, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    split = int(len(X_seq) * 0.8)
    return {
        'X_seq_train': X_seq[:split],
        'X_seq_test': X_seq[split:],
        'X_flat_train': X_flat[:split],
        'X_flat_test': X_flat[split:],
        'y_train': y_seq[:split],
        'y_test': y_seq[split:],
        'split': split,
        'scaler_y': scaler_y,
    }


def metrics_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    metrics = Evaluator.calculate_all_metrics(y_true, y_pred)
    mask = np.abs(y_true) > 0.1
    metrics['MAPE_filtered'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return metrics


def predict_arima(df, split):
    series = df[TARGET].astype(float).reset_index(drop=True)
    train_end = split + SEQ_LEN
    history = series.iloc[:train_end]
    test_series = series.iloc[train_end:]
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = ARIMA(history, order=(5, 1, 0)).fit()
        for actual in test_series:
            pred = float(result.forecast(steps=1).iloc[0])
            preds.append(pred)
            try:
                result = result.append([float(actual)], refit=False)
            except Exception:
                history = pd.concat([history, pd.Series([float(actual)])], ignore_index=True)
                result = ARIMA(history, order=(5, 1, 0)).fit()
    return np.asarray(preds, dtype=float)


def predict_svr(data):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(data['X_flat_train'], data['y_train'].ravel())
    pred_scaled = model.predict(data['X_flat_test']).reshape(-1, 1)
    return data['scaler_y'].inverse_transform(pred_scaled).flatten()


def predict_bilstm(data, epochs):
    set_seed()
    model = LSTMPredictor(input_size=len(FEATURES), hidden_size=64, num_layers=1, bidirectional=True, dropout=0.0)
    trainer = LSTMTrainer(model, lr=0.001)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(data['X_seq_train'], dtype=torch.float32),
            torch.tensor(data['y_train'], dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    trainer.train(train_loader, epochs=epochs)
    pred_scaled = trainer.predict(torch.tensor(data['X_seq_test'], dtype=torch.float32))
    return data['scaler_y'].inverse_transform(pred_scaled).flatten()


def run_one_well(well_file, epochs):
    df = prepare_dataframe(well_file)
    data = build_data(df)
    y_true = data['scaler_y'].inverse_transform(data['y_test']).flatten()

    predictions = {
        'ARIMA': predict_arima(df, data['split']),
        'SVR': predict_svr(data),
        'Bi-LSTM': predict_bilstm(data, epochs),
    }

    rows = []
    for model_name, y_pred in predictions.items():
        metrics = metrics_safe(y_true, y_pred)
        rows.append({
            'well_file': well_file,
            'model': model_name,
            'sample_count': len(y_true),
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'MAPE_filtered': metrics['MAPE_filtered'],
            'R2': metrics['R2'],
        })
    return rows


def main():
    set_seed()
    epochs = int(os.getenv('ALL_WELL_EPOCHS', '20'))
    limit = os.getenv('ALL_WELL_LIMIT')
    well_files = DL.get_all_well_files()
    if limit:
        well_files = well_files[:int(limit)]

    print('=' * 70)
    print('All-well ARIMA / SVR / Bi-LSTM comparison')
    print('=' * 70)
    print(f'[Config] wells={len(well_files)}, Bi-LSTM epochs={epochs}')

    all_rows = []
    failed_rows = []
    for idx, well_file in enumerate(well_files, start=1):
        print('\n' + '-' * 70)
        print(f'[Well {idx}/{len(well_files)}] {well_file}')
        try:
            rows = run_one_well(well_file, epochs)
            all_rows.extend(rows)
            for row in rows:
                print(f"  {row['model']:7s} RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, R2={row['R2']:.4f}, MAPE_filtered={row['MAPE_filtered']:.2f}%")
        except Exception as exc:
            print(f'  [Warn] failed: {exc}')
            failed_rows.append({'well_file': well_file, 'error': str(exc)})

    if not all_rows:
        raise RuntimeError('No well comparison results were generated.')

    detail_df = pd.DataFrame(all_rows)
    summary_df = detail_df.groupby('model', as_index=False).agg(
        wells=('well_file', 'nunique'),
        avg_RMSE=('RMSE', 'mean'),
        avg_MAE=('MAE', 'mean'),
        avg_R2=('R2', 'mean'),
        avg_MAPE_filtered=('MAPE_filtered', 'mean'),
        median_RMSE=('RMSE', 'median'),
        median_MAE=('MAE', 'median'),
        median_R2=('R2', 'median'),
    )
    failed_df = pd.DataFrame(failed_rows)

    detail_path = os.path.join(config.OUTPUT_DIR, 'all_well_model_comparison.xlsx')
    summary_path = os.path.join(config.OUTPUT_DIR, 'all_well_model_comparison_summary.xlsx')
    failed_path = os.path.join(config.OUTPUT_DIR, 'all_well_model_comparison_failed.xlsx')
    detail_df.to_excel(detail_path, index=False)
    summary_df.to_excel(summary_path, index=False)
    failed_df.to_excel(failed_path, index=False)

    print('\n' + '=' * 70)
    print('[Summary]')
    print(summary_df.to_string(index=False))
    print('\n[Saved]')
    print(f'  {detail_path}')
    print(f'  {summary_path}')
    print(f'  {failed_path}')


if __name__ == '__main__':
    main()
