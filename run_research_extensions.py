"""
Extended research experiments for the thesis.

Outputs:
- prediction_horizon_metrics.xlsx: 1/3/7-step model comparison
- feature_ablation_metrics.xlsx: feature engineering ablation
- cross_well_generalization_metrics.xlsx: pooled multi-well generalization
- residual_error_analysis.xlsx: residual and high-error interval analysis
- plots/*.png: paper-ready figures
"""
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

import config
from evaluation.metrics import Evaluator
from models.lstm_model import LSTMPredictor, LSTMTrainer
from pre_processing.cleaner import DataCleaner as DC
from pre_processing.data_loader import DataLoader as DL
from pre_processing.diagnosis_rules import DiagnosisRules
from pre_processing.feature_eng import FeatureEngineer as FE

warnings.filterwarnings("ignore")

TARGET = "gas_volume"
SEQ_LEN = 30
RANDOM_SEED = 42
BASE_FEATURES = ["wellhead_press", "gas_volume_lag_1", "wellhead_press_roll_7_mean"]
OUT_DIR = Path(config.OUTPUT_DIR) / "research_extensions"
PLOT_DIR = OUT_DIR / "plots"


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_font():
    for font_path in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "Microsoft YaHei", "SimHei"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_dataframe(well_file):
    df = DL.load_well_data(well_file)
    df = DC.unify_formats(df).sort_values("date").reset_index(drop=True)
    numeric = [c for c in ["wellhead_press", "casing_press", "prod_hours", "gas_volume"] if c in df.columns]
    df = DC.remove_outliers_3sigma(df, numeric)
    df = DC.handle_missing_values(df, numeric)
    df = FE.add_lagged_features(df, TARGET)
    df = FE.add_rolling_features(df, ["wellhead_press", "gas_volume"])
    df = df.dropna().reset_index(drop=True)
    return df


def available_features(df, requested):
    return [c for c in requested if c in df.columns and df[c].notna().any()]


def make_supervised(df, features, horizon=1, seq_len=SEQ_LEN):
    y_raw = df[TARGET].astype(float).values.reshape(-1, 1)
    x_raw = df[features].astype(float).values
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x_raw)
    y = scaler_y.fit_transform(y_raw)

    x_seq, x_flat, y_target, dates = [], [], [], []
    end = len(df) - horizon + 1
    for i in range(seq_len, end):
        target_idx = i + horizon - 1
        x_seq.append(x[i - seq_len:i])
        x_flat.append(x[i - 1])
        y_target.append(y[target_idx, 0])
        dates.append(df["date"].iloc[target_idx])

    x_seq = np.asarray(x_seq, dtype=np.float32)
    x_flat = np.asarray(x_flat, dtype=np.float32)
    y_target = np.asarray(y_target, dtype=np.float32).reshape(-1, 1)
    split = int(len(x_seq) * 0.8)
    return {
        "X_seq_train": x_seq[:split],
        "X_seq_test": x_seq[split:],
        "X_flat_train": x_flat[:split],
        "X_flat_test": x_flat[split:],
        "y_train": y_target[:split],
        "y_test": y_target[split:],
        "dates_test": np.asarray(dates[split:]),
        "split": split,
        "scaler_y": scaler_y,
    }


def metrics_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    m = Evaluator.calculate_all_metrics(y_true, y_pred)
    mask = np.abs(y_true) > 0.1
    m["MAPE_filtered"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    return m


def predict_arima_horizon(df, split, horizon):
    series = df[TARGET].astype(float).reset_index(drop=True)
    origins = list(range(SEQ_LEN + split, len(series) - horizon + 1))
    preds = []
    for idx in origins:
        history = series.iloc[:idx]
        result = ARIMA(history, order=(5, 1, 0)).fit()
        preds.append(float(result.forecast(steps=horizon).iloc[-1]))
    return np.asarray(preds, dtype=float)


def predict_svr(data):
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(data["X_flat_train"], data["y_train"].ravel())
    pred_scaled = model.predict(data["X_flat_test"]).reshape(-1, 1)
    return data["scaler_y"].inverse_transform(pred_scaled).flatten(), model


def predict_bilstm(data, input_size, epochs):
    set_seed()
    model = LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=1, bidirectional=True, dropout=0.0)
    trainer = LSTMTrainer(model, lr=0.001)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(data["X_seq_train"], dtype=torch.float32),
            torch.tensor(data["y_train"], dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    trainer.train(loader, epochs=epochs)
    pred_scaled = trainer.predict(torch.tensor(data["X_seq_test"], dtype=torch.float32))
    return data["scaler_y"].inverse_transform(pred_scaled).flatten(), model


def run_horizon_experiment(well_file, epochs):
    print("[Experiment] prediction horizons")
    df = prepare_dataframe(well_file)
    features = available_features(df, BASE_FEATURES)
    rows = []
    for horizon in [1, 3, 7]:
        data = make_supervised(df, features, horizon=horizon)
        y_true = data["scaler_y"].inverse_transform(data["y_test"]).flatten()
        predictions = {
            "ARIMA": predict_arima_horizon(df, data["split"], horizon),
            "SVR": predict_svr(data)[0],
            "Bi-LSTM": predict_bilstm(data, len(features), epochs)[0],
        }
        for model_name, pred in predictions.items():
            row = {"well_file": well_file, "horizon_days": horizon, "model": model_name}
            row.update(metrics_safe(y_true, pred))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_excel(OUT_DIR / "prediction_horizon_metrics.xlsx", index=False)
    return out


def run_feature_ablation(well_file, epochs):
    print("[Experiment] feature ablation")
    df = prepare_dataframe(well_file)
    groups = {
        "history_only": ["gas_volume_lag_1"],
        "pressure_plus_history": ["wellhead_press", "casing_press", "prod_hours", "gas_volume_lag_1"],
        "lag_features": ["gas_volume_lag_1", "gas_volume_lag_3", "gas_volume_lag_7"],
        "rolling_features": ["gas_volume_lag_1", "gas_volume_roll_3_mean", "gas_volume_roll_7_mean", "wellhead_press_roll_7_mean"],
        "full_features": [
            "wellhead_press", "casing_press", "prod_hours",
            "gas_volume_lag_1", "gas_volume_lag_3", "gas_volume_lag_7",
            "gas_volume_roll_3_mean", "gas_volume_roll_3_std",
            "gas_volume_roll_7_mean", "gas_volume_roll_7_std",
            "wellhead_press_roll_3_mean", "wellhead_press_roll_7_mean",
        ],
    }
    rows = []
    for group_name, requested in groups.items():
        features = available_features(df, requested)
        if not features:
            continue
        data = make_supervised(df, features, horizon=1)
        y_true = data["scaler_y"].inverse_transform(data["y_test"]).flatten()
        for model_name, pred in {
            "SVR": predict_svr(data)[0],
            "Bi-LSTM": predict_bilstm(data, len(features), epochs)[0],
        }.items():
            row = {
                "well_file": well_file,
                "feature_group": group_name,
                "feature_count": len(features),
                "features": ",".join(features),
                "model": model_name,
            }
            row.update(metrics_safe(y_true, pred))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_excel(OUT_DIR / "feature_ablation_metrics.xlsx", index=False)
    return out


def run_cross_well_generalization(epochs):
    print("[Experiment] pooled cross-well generalization")
    well_files = DL.get_all_well_files()
    features = None
    train_x, train_y, test_parts = [], [], []
    for well_file in well_files:
        try:
            df = prepare_dataframe(well_file)
            if features is None:
                features = available_features(df, BASE_FEATURES)
            data = make_supervised(df, features, horizon=1)
            if len(data["X_flat_train"]) < 20 or len(data["X_flat_test"]) < 5:
                continue
            train_x.append(data["X_flat_train"])
            train_y.append(data["y_train"])
            test_parts.append((well_file, data))
        except Exception as exc:
            print(f"  [Skip] {well_file}: {exc}")

    x_train = np.vstack(train_x)
    y_train = np.vstack(train_y).ravel()
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1).fit(x_train, y_train)

    x_seq_train = np.vstack([d["X_seq_train"] for _, d in test_parts])
    y_seq_train = np.vstack([d["y_train"] for _, d in test_parts])
    pooled_data = {
        "X_seq_train": x_seq_train,
        "y_train": y_seq_train,
        "X_seq_test": test_parts[0][1]["X_seq_test"],
        "scaler_y": test_parts[0][1]["scaler_y"],
    }
    model = LSTMPredictor(input_size=len(features), hidden_size=64, num_layers=1, bidirectional=True, dropout=0.0)
    trainer = LSTMTrainer(model, lr=0.001)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_seq_train, dtype=torch.float32), torch.tensor(y_seq_train, dtype=torch.float32)),
        batch_size=64,
        shuffle=True,
    )
    trainer.train(loader, epochs=epochs)
    torch.save(model.state_dict(), OUT_DIR / "pooled_bilstm_model.pth")

    rows = []
    for well_file, data in test_parts:
        y_true = data["scaler_y"].inverse_transform(data["y_test"]).flatten()
        svr_pred = data["scaler_y"].inverse_transform(svr.predict(data["X_flat_test"]).reshape(-1, 1)).flatten()
        lstm_scaled = trainer.predict(torch.tensor(data["X_seq_test"], dtype=torch.float32))
        lstm_pred = data["scaler_y"].inverse_transform(lstm_scaled).flatten()
        for model_name, pred in {"Pooled-SVR": svr_pred, "Pooled-Bi-LSTM": lstm_pred}.items():
            row = {"well_file": well_file, "model": model_name, "train_mode": "multi_well_pooled"}
            row.update(metrics_safe(y_true, pred))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_excel(OUT_DIR / "cross_well_generalization_metrics.xlsx", index=False)
    return out


def run_residual_analysis(well_file):
    print("[Experiment] residual and abnormal-condition analysis")
    pred_path = Path(config.OUTPUT_DIR) / "model_comparison_predictions.xlsx"
    pred_df = pd.read_excel(pred_path)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    rows = []
    for model in ["ARIMA", "SVR", "Bi-LSTM"]:
        residual = pred_df["actual"] - pred_df[model]
        abs_error = residual.abs()
        q90 = abs_error.quantile(0.90)
        for idx, item in pred_df.iterrows():
            rows.append({
                "date": item["date"],
                "model": model,
                "actual": item["actual"],
                "prediction": item[model],
                "residual": residual.iloc[idx],
                "abs_error": abs_error.iloc[idx],
                "is_high_error_top10pct": bool(abs_error.iloc[idx] >= q90),
            })
    residual_df = pd.DataFrame(rows)

    raw = DL.load_well_data(well_file)
    raw = DC.unify_formats(raw).sort_values("date")
    raw = DC.handle_missing_values(raw, [c for c in ["wellhead_press", "casing_press", "prod_hours", "gas_volume"] if c in raw.columns])
    diag = DiagnosisRules.apply_rules(raw)
    residual_df = residual_df.merge(diag[["date", "diag_label", "diag_state"]], on="date", how="left")
    residual_df.to_excel(OUT_DIR / "residual_error_analysis.xlsx", index=False)

    summary = residual_df.groupby(["model", "diag_state"], as_index=False).agg(
        samples=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        high_error_rate=("is_high_error_top10pct", "mean"),
    )
    summary.to_excel(OUT_DIR / "residual_diagnosis_summary.xlsx", index=False)
    return residual_df, summary


def plot_outputs(horizon_df, ablation_df, cross_df, residual_summary):
    sns.set_theme(style="whitegrid", palette="Set2")
    setup_font()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=horizon_df, x="horizon_days", y="RMSE", hue="model", marker="o", ax=ax)
    ax.set_title("不同预测步长下模型RMSE变化")
    ax.set_xlabel("预测步长/天")
    ax.set_ylabel("RMSE")
    fig.savefig(PLOT_DIR / "20_prediction_horizon_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=ablation_df, x="feature_group", y="RMSE", hue="model", ax=ax)
    ax.set_title("特征工程消融实验RMSE对比")
    ax.set_xlabel("特征组合")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=18)
    fig.savefig(PLOT_DIR / "21_feature_ablation_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=cross_df, x="model", y="RMSE", ax=ax)
    sns.stripplot(data=cross_df, x="model", y="RMSE", color="0.25", size=3, alpha=0.55, ax=ax)
    ax.set_title("多井合并训练的跨井泛化RMSE分布")
    ax.set_xlabel("模型")
    ax.set_ylabel("RMSE")
    fig.savefig(PLOT_DIR / "22_cross_well_generalization_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=residual_summary, x="diag_state", y="mean_abs_error", hue="model", ax=ax)
    ax.set_title("不同生产状态下的平均绝对误差")
    ax.set_xlabel("积液诊断状态")
    ax.set_ylabel("平均绝对误差")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(PLOT_DIR / "23_residual_by_diagnosis_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(horizon_df, ablation_df, cross_df, residual_summary):
    def table_text(df):
        return df.round(6).to_string(index=False)

    lines = []
    lines.append("# 研究扩展实验结果摘要")
    best_h = horizon_df.loc[horizon_df.groupby("horizon_days")["RMSE"].idxmin(), ["horizon_days", "model", "RMSE", "R2"]]
    lines.append("\n## 不同预测步长最优模型")
    lines.append(table_text(best_h))
    best_ab = ablation_df.loc[ablation_df.groupby(["feature_group"])["RMSE"].idxmin(), ["feature_group", "model", "RMSE", "R2"]]
    lines.append("\n## 特征消融最优结果")
    lines.append(table_text(best_ab))
    cross_summary = cross_df.groupby("model", as_index=False).agg(avg_RMSE=("RMSE", "mean"), avg_R2=("R2", "mean"), wells=("well_file", "nunique"))
    lines.append("\n## 跨井泛化平均结果")
    lines.append(table_text(cross_summary))
    lines.append("\n## 异常状态误差分析")
    lines.append(table_text(residual_summary))
    (OUT_DIR / "research_extensions_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    set_seed()
    ensure_dirs()
    well_files = sorted(DL.get_all_well_files())
    if not well_files:
        raise RuntimeError("No well data found.")
    well_file = os.getenv("RESEARCH_WELL", well_files[0])
    epochs = int(os.getenv("RESEARCH_EPOCHS", "10"))
    print("=" * 70)
    print("Research extension experiments")
    print(f"Well: {well_file}")
    print(f"Bi-LSTM epochs: {epochs}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    horizon_df = run_horizon_experiment(well_file, epochs)
    ablation_df = run_feature_ablation(well_file, epochs)
    cross_df = run_cross_well_generalization(epochs)
    _, residual_summary = run_residual_analysis(well_file)
    plot_outputs(horizon_df, ablation_df, cross_df, residual_summary)
    write_summary(horizon_df, ablation_df, cross_df, residual_summary)
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
