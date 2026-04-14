import os
import sys

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

FEATURES = ['wellhead_press', 'gas_volume_lag_1', 'wellhead_press_roll_7_mean']
TARGET = 'gas_volume'
SEQ_LEN = 30
MODEL_PATH = os.path.join(config.OUTPUT_DIR, 'ultra_lstm_v2.pth')

st.set_page_config(page_title='智慧油气田生产预测与诊断系统', layout='wide')


@st.cache_data
def get_data(well_name):
    df = DL.load_well_data(well_name)
    df = DC.unify_formats(df)
    df_diag = DR.apply_rules(df)
    return df_diag


@st.cache_resource
def load_lstm_model():
    if not os.path.exists(MODEL_PATH):
        return None, f'模型文件不存在: {MODEL_PATH}'

    model = LSTMPredictor(input_size=len(FEATURES), hidden_size=128, num_layers=2, bidirectional=True)
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except Exception as exc:
        return None, f'模型加载失败: {exc}'


def predict_with_lstm(df):
    model, error = load_lstm_model()
    if error:
        return None, None, error

    work_df = df.copy()
    work_df = DC.remove_outliers_3sigma(work_df, ['wellhead_press', 'gas_volume'])
    work_df = DC.handle_missing_values(work_df, ['wellhead_press', 'gas_volume'])
    work_df = FE.add_lagged_features(work_df, TARGET)
    work_df = FE.add_rolling_features(work_df, ['wellhead_press'])
    work_df = work_df.dropna().copy()

    missing_features = [col for col in FEATURES if col not in work_df.columns]
    if missing_features:
        return None, None, f'缺少预测特征: {missing_features}'
    if len(work_df) <= SEQ_LEN:
        return None, None, '可用数据不足，无法构造 LSTM 时序窗口。'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(work_df[FEATURES].values)
    y_scaled = scaler_y.fit_transform(work_df[TARGET].values.reshape(-1, 1))

    X_seq = [X_scaled[i:i + SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_scaled[SEQ_LEN:]).flatten()
    pred_df = pd.DataFrame({
        'date': work_df['date'].iloc[SEQ_LEN:].values,
        '真实产量': y_true,
        'PSO-LSTM预测产量': y_pred,
    })
    return pred_df, y_pred, None


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


st.sidebar.title('系统控制面板')
st.sidebar.markdown('---')
well_files = DL.get_all_well_files()
selected_well = st.sidebar.selectbox('选择目标井号', well_files)
selected_model = st.sidebar.radio('选择预测模型', ['PSO-LSTM (真实模型权重)'])
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
pred_df, pred_vals, pred_error = predict_with_lstm(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['gas_volume'], name='真实产量', line=dict(color='royalblue', width=2)))
if pred_df is not None:
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['PSO-LSTM预测产量'],
        name=f'{selected_model} 预测',
        line=dict(color='firebrick', width=2, dash='dot'),
    ))
else:
    st.warning(pred_error)
fig.update_layout(title='产量历史与模型预测对比', xaxis_title='日期', yaxis_title='日产量 (10^4 m3)')
st.plotly_chart(fig, use_container_width=True)

c_left, c_right = st.columns([2, 1])
with c_left:
    st.markdown('### 积液风险诊断日志')
    diag_df = df[['date', 'wellhead_press', 'gas_volume', 'diag_state']].tail(10).astype(str)
    st.dataframe(diag_df, use_container_width=True)

with c_right:
    st.markdown('### 当前井模型精度')
    if pred_df is not None:
        metrics = calculate_metrics(pred_df['真实产量'].values, pred_df['PSO-LSTM预测产量'].values)
        metrics_data = {
            '指标': ['RMSE', 'MAE', 'MAPE', 'R2'],
            'PSO-LSTM': [
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

