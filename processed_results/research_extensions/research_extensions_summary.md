# 研究扩展实验结果摘要

## 不同预测步长最优模型
 horizon_days   model     RMSE       R2
            1   ARIMA 0.122470 0.744576
            3     SVR 0.178493 0.457437
            7 Bi-LSTM 0.225481 0.126758

## 特征消融最优结果
        feature_group model     RMSE       R2
        full_features   SVR 0.138890 0.671492
         history_only   SVR 0.153184 0.600395
         lag_features   SVR 0.150645 0.613530
pressure_plus_history   SVR 0.131691 0.704662
     rolling_features   SVR 0.125523 0.731680

## 跨井泛化平均结果
         model  avg_RMSE   avg_R2  wells
Pooled-Bi-LSTM  0.225717 0.421972     30
    Pooled-SVR  0.227244 0.415908     30

## 异常状态误差分析
  model    diag_state  samples  mean_abs_error  high_error_rate
  ARIMA  Accumulation       40        0.084725         0.250000
  ARIMA        Closed       30        0.028478         0.033333
  ARIMA WorkWithWater      101        0.050046         0.069307
Bi-LSTM  Accumulation       40        0.093147         0.125000
Bi-LSTM        Closed       30        0.103060         0.233333
Bi-LSTM WorkWithWater      101        0.071771         0.059406
    SVR  Accumulation       40        0.076554         0.175000
    SVR        Closed       30        0.084537         0.233333
    SVR WorkWithWater      101        0.034788         0.039604