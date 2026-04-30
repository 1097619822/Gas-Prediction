import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

import config
from models.classifier import LSTMClassifier, ClassificationTrainer
from pre_processing.data_loader import DataLoader as DL
from pre_processing.diagnosis_rules import DiagnosisRules as DR

STATE_MAP = {0: 'Normal', 1: 'WorkWithWater', 2: 'Accumulation', 3: 'Closed'}
FEATURES = ['wellhead_press', 'gas_volume']
SEQ_LEN = 7


def prepare_sequences(X, y, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long).view(-1)
    return X_tensor, y_tensor


def evaluate_classifier(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_test.cpu().numpy()

    labels = [0, 1, 2, 3]
    target_names = [STATE_MAP[label] for label in labels]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return acc, cm, report_dict


def save_evaluation_outputs(acc, cm, report_dict):
    labels = [STATE_MAP[i] for i in range(4)]
    cm_df = pd.DataFrame(cm, index=[f'true_{x}' for x in labels], columns=[f'pred_{x}' for x in labels])
    report_df = pd.DataFrame(report_dict).T

    cm_path = os.path.join(config.OUTPUT_DIR, 'diagnosis_confusion_matrix.xlsx')
    report_path = os.path.join(config.OUTPUT_DIR, 'diagnosis_classification_report.xlsx')
    summary_path = os.path.join(config.OUTPUT_DIR, 'diagnosis_metrics_summary.xlsx')

    cm_df.to_excel(cm_path)
    report_df.to_excel(report_path)
    pd.DataFrame([{'accuracy': acc}]).to_excel(summary_path, index=False)
    return cm_path, report_path, summary_path


def main():
    print('=' * 70)
    print('Liquid accumulation diagnosis model training')
    print('=' * 70)

    well_files = DL.get_all_well_files()
    all_x, all_y = [], []
    print(f'[Step 1] Scanning {len(well_files)} well files...')

    for filename in well_files:
        try:
            df = DL.load_well_data(filename)
            df_labeled = DR.apply_rules(df)
            df_valid = df_labeled[df_labeled['diag_label'] >= 0].copy()
            if len(df_valid) < 20:
                continue

            X_raw = df_valid[FEATURES].values
            X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
            X_norm = (X_raw - X_min) / (X_max - X_min + 1e-6)
            y_raw = df_valid['diag_label'].values
            tx, ty = prepare_sequences(X_norm, y_raw, seq_len=SEQ_LEN)
            if len(tx) > 0:
                all_x.append(tx)
                all_y.append(ty)
        except Exception as exc:
            print(f'[Warn] Skip {filename}: {exc}')

    if not all_x:
        print('\n[Fatal] No diagnosis samples were built.')
        return

    X_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)

    print('\n[Dataset]')
    print(f' - total samples: {len(X_total)}')
    unique, counts = np.unique(y_total.numpy(), return_counts=True)
    for u, c in zip(unique, counts):
        print(f' - {STATE_MAP[int(u)]:14s}: {c}')

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(X_total), generator=generator)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X_total[train_idx], y_total[train_idx]
    X_test, y_test = X_total[test_idx], y_total[test_idx]

    print(f' - train samples: {len(X_train)}')
    print(f' - test samples: {len(X_test)}')

    print('\n[Step 2] Training Bi-LSTM diagnosis classifier...')
    model = LSTMClassifier(input_size=2, num_classes=4)
    trainer = ClassificationTrainer(model, lr=0.002)
    epochs = int(os.getenv('DIAG_EPOCHS', '30'))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    trainer.train_step(train_loader, None, epochs=epochs)

    print('\n[Step 3] Evaluating classifier...')
    acc, cm, report_dict = evaluate_classifier(model, X_test, y_test)
    print(f' - accuracy: {acc:.4f}')
    print(' - confusion matrix:')
    print(cm)

    save_path = os.path.join(config.OUTPUT_DIR, 'final_diagnosis_model.pth')
    torch.save(model.state_dict(), save_path)
    cm_path, report_path, summary_path = save_evaluation_outputs(acc, cm, report_dict)

    print('\n[Saved]')
    print(f' - model: {save_path}')
    print(f' - confusion matrix: {cm_path}')
    print(f' - classification report: {report_path}')
    print(f' - metrics summary: {summary_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
