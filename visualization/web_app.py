import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from models.lstm_model import LSTMPredictor
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.data_loader import DataLoader as DL
from pre_processing.diagnosis_rules import DiagnosisRules as DR
from pre_processing.feature_eng import FeatureEngineer as FE

DEFAULT_FEATURES = ['wellhead_press', 'gas_volume_lag_1', 'wellhead_press_roll_7_mean']
DEFAULT_TARGET = 'gas_volume'
DEFAULT_SEQ_LEN = 30
LSTM_MODEL_PATH = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2.pth')
LSTM_CHECKPOINT_PATH = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2_checkpoint.pth')
COMPARISON_CHECKPOINT_PATH = os.path.join(config.OUTPUT_DIR, 'model_comparison_checkpoint.pth')

st.set_page_config(page_title='智慧油气田生产预测与诊断系统', layout='wide')


@st.cache_data
def get_data(well_name):
    df = DL.load_well_data(well_name)
    df = DC.unify_formats(df)
    df_diag = DR.apply_rules(df)
    return df_diag


def prepare_features(df, features, target):
    work_df = df.copy()
    work_df = DC.remove_outliers_3sigma(work_df, ['wellhead_press', 'gas_volume'])
    work_df = DC.handle_missing_values(work_df, ['wellhead_press', 'gas_volume'])
    work_df = FE.add_lagged_features(work_df, target)
    work_df = FE.add_rolling_features(work_df, ['wellhead_press'])
    work_df = work_df.dropna().copy()
    missing_features = [col for col in features if col not in work_df.columns]
    if missing_features:
        raise ValueError(f'缺少预测特征: {missing_features}')
    return work_df


@st.cache_resource
def load_comparison_bundle():
    if not os.path.exists(COMPARISON_CHECKPOINT_PATH):
        return None, f'对比实验 checkpoint 不存在: {COMPARISON_CHECKPOINT_PATH}'
    try:
        checkpoint = torch.load(COMPARISON_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        features = checkpoint.get('features', DEFAULT_FEATURES)
        model_config = checkpoint.get('model_config', {})
        bilstm = LSTMPredictor(
            input_size=model_config.get('input_size', len(features)),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            bidirectional=model_config.get('bidirectional', True),
            dropout=model_config.get('dropout', 0.2),
        )
        bilstm.load_state_dict(checkpoint['bilstm_state_dict'])
        bilstm.eval()
        bundle = {
            'features': features,
            'target': checkpoint.get('target', DEFAULT_TARGET),
            'seq_len': checkpoint.get('seq_len', DEFAULT_SEQ_LEN),
            'arima': joblib.load(checkpoint['arima_model_path']),
            'svr': joblib.load(checkpoint['svr_model_path']),
            'bilstm': bilstm,
            'scaler_x': joblib.load(checkpoint['scaler_x_path']),
            'scaler_y': joblib.load(checkpoint['scaler_y_path']),
            'metrics': checkpoint.get('metrics', {}),
            'predictions_path': checkpoint.get('predictions_path'),
            'well_file': checkpoint.get('well_file'),
            'source': COMPARISON_CHECKPOINT_PATH,
        }
        return bundle, None
    except Exception as exc:
        return None, f'对比实验模型加载失败: {exc}'


@st.cache_resource
def load_lstm_fallback_bundle():
    if os.path.exists(LSTM_CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(LSTM_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            features = checkpoint.get('features', DEFAULT_FEATURES)
            model_config = checkpoint.get('model_config', {})
            model = LSTMPredictor(
                input_size=model_config.get('input_size', len(features)),
                hidden_size=model_config.get('hidden_size', 128),
                num_layers=model_config.get('num_layers', 2),
                bidirectional=model_config.get('bidirectional', True),
                dropout=model_config.get('dropout', 0.2),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            scaler_x = joblib.load(checkpoint['scaler_x_path']) if os.path.exists(checkpoint['scaler_x_path']) else None
            scaler_y = joblib.load(checkpoint['scaler_y_path']) if os.path.exists(checkpoint['scaler_y_path']) else None
            return {
                'features': features,
                'target': checkpoint.get('target', DEFAULT_TARGET),
                'seq_len': checkpoint.get('seq_len', DEFAULT_SEQ_LEN),
                'bilstm': model,
                'scaler_x': scaler_x,
                'scaler_y': scaler_y,
                'metrics': {'Bi-LSTM': checkpoint.get('metrics', {})},
                'well_file': checkpoint.get('target_well'),
                'source': LSTM_CHECKPOINT_PATH,
            }, None
        except Exception as exc:
            return None, f'Bi-LSTM checkpoint 加载失败: {exc}'

    if os.path.exists(LSTM_MODEL_PATH):
        model = LSTMPredictor(input_size=len(DEFAULT_FEATURES), hidden_size=128, num_layers=2, bidirectional=True)
        model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location='cpu'))
        model.eval()
        return {
            'features': DEFAULT_FEATURES,
            'target': DEFAULT_TARGET,
            'seq_len': DEFAULT_SEQ_LEN,
            'bilstm': model,
            'scaler_x': None,
            'scaler_y': None,
            'metrics': {},
            'well_file': None,
            'source': LSTM_MODEL_PATH,
        }, None
    return None, f'未找到 Bi-LSTM 模型文件: {LSTM_CHECKPOINT_PATH}'


def get_model_bundle():
    bundle, error = load_comparison_bundle()
    if bundle is not None:
        return bundle, None
    return load_lstm_fallback_bundle()


def predict_selected_model(df, model_name):
    bundle, error = get_model_bundle()
    if error:
        return None, None, error

    features = bundle['features']
    target = bundle['target']
    seq_len = bundle['seq_len']
    work_df = prepare_features(df, features, target)

    if len(work_df) <= seq_len:
        return None, bundle, '可用数据不足，无法构造预测窗口。'

    if model_name == 'ARIMA' and bundle.get('predictions_path') and os.path.exists(bundle['predictions_path']):
        saved_pred_df = pd.read_excel(bundle['predictions_path'])
        y_pred = saved_pred_df['ARIMA'].to_numpy(dtype=float)
        pred_dates = pd.to_datetime(saved_pred_df['date']).values
        y_true = saved_pred_df['actual'].to_numpy(dtype=float)
    elif model_name == 'ARIMA' and 'arima' in bundle:
        steps = len(work_df) - seq_len
        y_pred = np.asarray(bundle['arima'].forecast(steps=steps), dtype=float)
        pred_dates = work_df['date'].iloc[seq_len:].values
        y_true = work_df[target].iloc[seq_len:].to_numpy(dtype=float)
    else:
        scaler_x = bundle.get('scaler_x') or StandardScaler().fit(work_df[features].values)
        scaler_y = bundle.get('scaler_y') or StandardScaler().fit(work_df[target].values.reshape(-1, 1))
        X_scaled = scaler_x.transform(work_df[features].values)
        X_seq = [X_scaled[i:i + seq_len] for i in range(len(X_scaled) - seq_len)]
        pred_dates = work_df['date'].iloc[seq_len:].values
        y_true = work_df[target].iloc[seq_len:].to_numpy(dtype=float)

        if model_name == 'SVR' and 'svr' in bundle:
            X_flat = np.array([X_scaled[i + seq_len] for i in range(len(X_scaled) - seq_len)], dtype=np.float32)
            y_pred_scaled = bundle['svr'].predict(X_flat).reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        else:
            X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
            with torch.no_grad():
                y_pred_scaled = bundle['bilstm'](X_tensor).numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    n = min(len(y_true), len(y_pred), len(pred_dates))
    pred_df = pd.DataFrame({
        'date': pred_dates[:n],
        '真实产量': y_true[:n],
        f'{model_name}预测产量': y_pred[:n],
    })
    return pred_df, bundle, None


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = np.abs(y_true) > 0.1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


st.sidebar.title('系统控制面板')
st.sidebar.markdown('---')
well_files = DL.get_all_well_files()
selected_well = st.sidebar.selectbox('选择目标井号', well_files)
selected_model = st.sidebar.radio('选择预测模型', ['ARIMA', 'SVR', 'Bi-LSTM'])
st.sidebar.markdown('---')
st.sidebar.info('系统基于油田生产 Excel 数据构建，集成产量预测与积液诊断。')

st.title('智慧油气田生产预测与诊断演示系统')
st.subheader(f'当前监测井: {selected_well}')

df = get_data(selected_well)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('实时油压 (MPa)', f"{df['wellhead_press'].iloc[-1]:.2f}")
with col2:
    st.metric('实时套压 (MPa)', f"{df['casing_press'].iloc[-1]:.2f}")
with col3:
    st.metric('当日产量 (10^4 m3)', f"{df['gas_volume'].iloc[-1]:.2f}")
with col4:
    state = df['diag_state'].iloc[-1]
    color = 'green' if state == 'Normal' else 'orange' if state == 'WorkWithWater' else 'red'
    st.markdown(f'**当前状态** :{color}[{state}]')

st.markdown('### 产量回归预测对比')
pred_df, bundle, pred_error = predict_selected_model(df, selected_model)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['gas_volume'], name='真实产量', line=dict(color='royalblue', width=2)))
if pred_df is not None:
    pred_col = f'{selected_model}预测产量'
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df[pred_col],
        name=f'{selected_model} 预测',
        line=dict(color='firebrick', width=2, dash='dot'),
    ))
else:
    st.warning(pred_error)
fig.update_layout(title='产量历史与模型预测对比', xaxis_title='日期', yaxis_title='日产量 (10^4 m3)')
st.plotly_chart(fig, width='stretch')

if bundle is not None:
    st.caption(f"模型来源: {bundle['source']} | 训练井: {bundle.get('well_file') or '未知'}")

c_left, c_right = st.columns([2, 1])
with c_left:
    st.markdown('### 积液风险诊断日志')
    diag_df = df[['date', 'wellhead_press', 'gas_volume', 'diag_state']].tail(10).astype(str)
    st.dataframe(diag_df, use_container_width=True)

with c_right:
    st.markdown('### 当前井模型精度')
    if pred_df is not None:
        pred_col = f'{selected_model}预测产量'
        metrics = calculate_metrics(pred_df['真实产量'].values, pred_df[pred_col].values)
        metrics_data = {
            '指标': ['RMSE', 'MAE', 'MAPE(过滤低产量)', 'R2'],
            selected_model: [
                f"{metrics['RMSE']:.4f}",
                f"{metrics['MAE']:.4f}",
                f"{metrics['MAPE']:.2f}%",
                f"{metrics['R2']:.4f}",
            ],
        }
        st.table(pd.DataFrame(metrics_data))
    else:
        st.info('模型预测不可用，无法计算当前井指标。')

st.markdown('---')
if st.button('导出当前井诊断报告'):
    report_path = os.path.join(config.OUTPUT_DIR, f'{os.path.splitext(selected_well)[0]}_report.csv')
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    st.success(f'报表已生成: {report_path}')

