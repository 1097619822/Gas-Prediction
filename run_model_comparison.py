"""
Run data-cleaning statistics and model comparison experiments.

Models:
- ARIMA: traditional time-series baseline
- SVR: machine-learning baseline
- Bi-LSTM: deep-learning model

Outputs are saved under processed_results/ for paper tables and Web reuse.
"""
import os
import sys
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
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


def prepare_well_dataframe(well_file):
    raw_df = DL.load_well_data(well_file)
    raw_df = DC.unify_formats(raw_df)

    numeric_cols = ['wellhead_press', 'gas_volume']
    missing_before = raw_df[numeric_cols].isna().sum()
    zero_like_before = (raw_df[numeric_cols] == 0).sum()

    cleaned_df = DC.remove_outliers_3sigma(raw_df, numeric_cols)
    missing_after_outlier = cleaned_df[numeric_cols].isna().sum()
    outliers_removed = (missing_after_outlier - missing_before).clip(lower=0)
    filled_df = DC.handle_missing_values(cleaned_df, numeric_cols)
    missing_after = filled_df[numeric_cols].isna().sum()

    feat_df = FE.add_lagged_features(filled_df, TARGET)
    feat_df = FE.add_rolling_features(feat_df, ['wellhead_press'])
    before_dropna = len(feat_df)
    feat_df = feat_df.dropna().copy()

    cleaning_stats = {
        'well_file': well_file,
        'raw_rows': len(raw_df),
        'feature_rows_before_dropna': before_dropna,
        'final_rows': len(feat_df),
        'missing_before_wellhead_press': int(missing_before.get('wellhead_press', 0)),
        'missing_before_gas_volume': int(missing_before.get('gas_volume', 0)),
        'zero_wellhead_press': int(zero_like_before.get('wellhead_press', 0)),
        'zero_gas_volume': int(zero_like_before.get('gas_volume', 0)),
        'outliers_wellhead_press_3sigma': int(outliers_removed.get('wellhead_press', 0)),
        'outliers_gas_volume_3sigma': int(outliers_removed.get('gas_volume', 0)),
        'missing_after_wellhead_press': int(missing_after.get('wellhead_press', 0)),
        'missing_after_gas_volume': int(missing_after.get('gas_volume', 0)),
        'dropped_rows_by_feature_engineering': int(before_dropna - len(feat_df)),
    }
    return feat_df, cleaning_stats


def build_supervised_data(df):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df[FEATURES].values)
    y_scaled = scaler_y.fit_transform(df[TARGET].values.reshape(-1, 1))

    X_seq, y_seq = [], []
    X_flat, y_flat = [], []
    dates = []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        X_flat.append(X_scaled[i + SEQ_LEN])
        y_seq.append(y_scaled[i + SEQ_LEN])
        y_flat.append(y_scaled[i + SEQ_LEN, 0])
        dates.append(df['date'].iloc[i + SEQ_LEN])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    X_flat = np.array(X_flat, dtype=np.float32)
    y_flat = np.array(y_flat, dtype=np.float32)
    dates = np.array(dates)

    split = int(len(X_seq) * 0.8)
    data = {
        'X_seq_train': X_seq[:split],
        'y_seq_train': y_seq[:split],
        'X_seq_test': X_seq[split:],
        'y_seq_test': y_seq[split:],
        'X_flat_train': X_flat[:split],
        'y_flat_train': y_flat[:split],
        'X_flat_test': X_flat[split:],
        'y_flat_test': y_flat[split:],
        'dates_test': dates[split:],
        'split': split,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
    }
    return data


def calculate_metrics_safe(y_true, y_pred):
    metrics = Evaluator.calculate_all_metrics(y_true, y_pred)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.abs(y_true) > 0.1
    filtered_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    metrics['MAPE_filtered'] = filtered_mape
    return metrics


def train_arima(df, split_index):
    series = df[TARGET].astype(float).reset_index(drop=True)
    train_end = split_index + SEQ_LEN
    train_series = series.iloc[:train_end]
    test_series = series.iloc[train_end:]
    history = train_series.tolist()
    rolling_forecast = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_fit = ARIMA(train_series, order=(5, 1, 0)).fit()

        for actual in test_series:
            step_model = ARIMA(history, order=(5, 1, 0)).fit()
            pred = float(step_model.forecast(steps=1)[0])
            rolling_forecast.append(pred)
            # Rolling one-step evaluation uses the observed value as the next history point.
            history.append(float(actual))

    return model_fit, np.asarray(rolling_forecast, dtype=float)


def train_svr(data):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(data['X_flat_train'], data['y_flat_train'])
    y_pred_scaled = model.predict(data['X_flat_test']).reshape(-1, 1)
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled).flatten()
    return model, y_pred


def train_bilstm(data, epochs):
    model = LSTMPredictor(input_size=len(FEATURES), hidden_size=64, num_layers=1, bidirectional=True, dropout=0.0)
    trainer = LSTMTrainer(model, lr=0.001)
    X_train = torch.tensor(data['X_seq_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_seq_train'], dtype=torch.float32)
    X_test = torch.tensor(data['X_seq_test'], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    trainer.train(train_loader, epochs=epochs)

    y_pred_scaled = trainer.predict(X_test)
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled).flatten()
    return model, y_pred


def save_outputs(well_file, df, cleaning_stats, data, models, predictions, metrics, epochs):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    cleaning_df = pd.DataFrame([cleaning_stats])
    cleaning_path = os.path.join(config.OUTPUT_DIR, 'data_cleaning_stats.xlsx')
    cleaning_df.to_excel(cleaning_path, index=False)

    metrics_rows = []
    for model_name, model_metrics in metrics.items():
        row = {'model': model_name}
        row.update(model_metrics)
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(config.OUTPUT_DIR, 'model_comparison_metrics.xlsx')
    metrics_df.to_excel(metrics_path, index=False)

    pred_df = pd.DataFrame({
        'date': data['dates_test'],
        'actual': data['scaler_y'].inverse_transform(data['y_seq_test']).flatten(),
        'ARIMA': predictions['ARIMA'],
        'SVR': predictions['SVR'],
        'Bi-LSTM': predictions['Bi-LSTM'],
    })
    pred_path = os.path.join(config.OUTPUT_DIR, 'model_comparison_predictions.xlsx')
    pred_df.to_excel(pred_path, index=False)

    arima_path = os.path.join(config.OUTPUT_DIR, 'arima_model.pkl')
    svr_path = os.path.join(config.OUTPUT_DIR, 'svr_model.pkl')
    bilstm_weight_path = os.path.join(config.OUTPUT_DIR, 'bilstm_comparison_model.pth')
    comparison_checkpoint_path = os.path.join(config.OUTPUT_DIR, 'model_comparison_checkpoint.pth')
    scaler_x_path = os.path.join(config.OUTPUT_DIR, 'model_comparison_scaler_x.pkl')
    scaler_y_path = os.path.join(config.OUTPUT_DIR, 'model_comparison_scaler_y.pkl')

    joblib.dump(models['ARIMA'], arima_path)
    joblib.dump(models['SVR'], svr_path)
    torch.save(models['Bi-LSTM'].state_dict(), bilstm_weight_path)
    joblib.dump(data['scaler_x'], scaler_x_path)
    joblib.dump(data['scaler_y'], scaler_y_path)

    checkpoint = {
        'well_file': well_file,
        'features': FEATURES,
        'target': TARGET,
        'seq_len': SEQ_LEN,
        'split': data['split'],
        'epochs': epochs,
        'model_config': {
            'input_size': len(FEATURES),
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.0,
        },
        'bilstm_state_dict': models['Bi-LSTM'].state_dict(),
        'arima_model_path': arima_path,
        'svr_model_path': svr_path,
        'bilstm_weight_path': bilstm_weight_path,
        'scaler_x_path': scaler_x_path,
        'scaler_y_path': scaler_y_path,
        'metrics_path': metrics_path,
        'predictions_path': pred_path,
        'cleaning_stats_path': cleaning_path,
        'metrics': metrics,
        'arima_mode': 'rolling_one_step_observed_history',
    }
    torch.save(checkpoint, comparison_checkpoint_path)

    print('\n[Saved]')
    for path in [cleaning_path, metrics_path, pred_path, arima_path, svr_path, bilstm_weight_path, comparison_checkpoint_path, scaler_x_path, scaler_y_path]:
        print(f'  {path}')


def main():
    set_seed()
    well_files = DL.get_all_well_files()
    if not well_files:
        raise RuntimeError(f'No Excel files found in {config.DATA_DIR}')

    well_file = os.getenv('COMPARISON_WELL', well_files[0])
    epochs = int(os.getenv('COMPARISON_EPOCHS', '50'))
    print('=' * 70)
    print('Data cleaning stats and model comparison experiment')
    print('=' * 70)
    print(f'[Data] Well: {well_file}')
    print(f'[Train] Bi-LSTM epochs: {epochs}')

    df, cleaning_stats = prepare_well_dataframe(well_file)
    data = build_supervised_data(df)
    y_true = data['scaler_y'].inverse_transform(data['y_seq_test']).flatten()

    print('[Model] Training ARIMA...')
    arima_model, arima_pred = train_arima(df, data['split'])

    print('[Model] Training SVR...')
    svr_model, svr_pred = train_svr(data)

    print('[Model] Training Bi-LSTM...')
    bilstm_model, bilstm_pred = train_bilstm(data, epochs)

    predictions = {
        'ARIMA': arima_pred[:len(y_true)],
        'SVR': svr_pred[:len(y_true)],
        'Bi-LSTM': bilstm_pred[:len(y_true)],
    }
    y_true = y_true[:min(len(y_true), *(len(v) for v in predictions.values()))]
    for key in predictions:
        predictions[key] = predictions[key][:len(y_true)]

    metrics = {name: calculate_metrics_safe(y_true, pred) for name, pred in predictions.items()}
    print('\n[Metrics]')
    for name, model_metrics in metrics.items():
        print(f'  {name}: RMSE={model_metrics["RMSE"]:.4f}, MAE={model_metrics["MAE"]:.4f}, R2={model_metrics["R2"]:.4f}, MAPE_filtered={model_metrics["MAPE_filtered"]:.2f}%')

    save_outputs(well_file, df, cleaning_stats, data, {'ARIMA': arima_model, 'SVR': svr_model, 'Bi-LSTM': bilstm_model}, predictions, metrics, epochs)
    print('\nDone.')


if __name__ == '__main__':
    main()


