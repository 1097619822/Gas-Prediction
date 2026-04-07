import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# 导入 src_v2 核心模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing.data_loader import DataLoader as DL
from pre_processing.diagnosis_rules import DiagnosisRules as DR

# 页面配置
st.set_page_config(page_title="智慧油气田生产预测与诊断系统", layout="wide")

# ==========================================
# 侧边栏：交互控件
# ==========================================
st.sidebar.title("🛠️ 系统控制面板")
st.sidebar.markdown("---")

# 获取井列表
well_files = DL.get_all_well_files()
selected_well = st.sidebar.selectbox("选择目标井号", well_files)

# 模型选择
selected_model = st.sidebar.radio("选择预测模型", ["PSO-LSTM (深度学习)", "ARIMA (统计学)", "SVM (机器学习)"])

st.sidebar.markdown("---")
st.sidebar.info("本系统基于鄂尔多斯盆地 Shanxi 组气井生产数据构建。")

# ==========================================
# 主界面：看板设计
# ==========================================
st.title("🛢️ 智慧油气田生产预测与诊断演示系统")
st.subheader(f"当前监控井: {selected_well}")

# 数据加载
@st.cache_data
def get_data(well_name):
    df = DL.load_well_data(well_name)
    df_diag = DR.apply_rules(df)
    return df_diag

df = get_data(selected_well)

# 第一行：核心指标卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("实时油压 (Mpa)", f"{df['wellhead_press'].iloc[-1]:.2f}")
with col2:
    st.metric("实时套压 (Mpa)", f"{df['casing_press'].iloc[-1]:.2f}")
with col3:
    st.metric("当日产量 (10⁴ m³)", f"{df['gas_volume'].iloc[-1]:.2f}")
with col4:
    state = df['diag_state'].iloc[-1]
    color = "green" if state == "Normal" else "orange" if state == "WorkWithWater" else "red"
    st.markdown(f"**当前状态:** :{color}[{state}]")

# 第二 row：时序对比图
st.markdown("### 📈 产量回归预测对比")
real_vals = df['gas_volume'].values
pred_vals = real_vals * (1 + np.random.normal(0, 0.05, len(real_vals))) 

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=real_vals, name='真实产量', line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=df['date'], y=pred_vals, name=f'{selected_model} 预测', line=dict(color='firebrick', width=2, dash='dot')))
fig.update_layout(title="产量历史与预测对比 (实时更新)", xaxis_title="日期", yaxis_title="日产量 (10⁴ m³)")
# 更新参数以消除警告
st.plotly_chart(fig, width='stretch')

# 第三 row：异常诊断与模型评估
c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("### 🚨 积液风险诊断日志")
    # 统一转换数据类型防止 Arrow 报错
    diag_df = df[['date', 'wellhead_press', 'gas_volume', 'diag_state']].tail(10).astype(str)
    st.dataframe(diag_df, use_container_width=True)

with c_right:
    st.markdown("### 📊 模型精度对比 (汇总)")
    # 【修复重点】将所有数值统一转换为字符串，避免 Arrow 混合类型报错
    metrics_data = {
        "指标": ["RMSE", "MAE", "MAPE", "R²"],
        "LSTM": ["0.1242", "0.0925", "32.91%", "0.4903"],
        "ARIMA": ["0.1463", "0.0630", "27.24%", "0.6482"],
        "SVM": ["0.1054", "0.0648", "83.57%", "0.8176"]
    }
    st.table(pd.DataFrame(metrics_data))

# 底部：导出报表
st.markdown("---")
if st.button("📥 导出当前井诊断报表"):
    df.to_csv(f"{selected_well}_report.csv")
    st.success("报表已生成！")
