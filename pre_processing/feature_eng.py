import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class FeatureEngineer:
    """
    特征工程：滞后、窗口、归一化。
    """
    @staticmethod
    def add_lagged_features(df, target_col='gas_volume'):
        df_lagged = df.copy()
        for lag in config.LAG_DAYS:
            df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)
        return df_lagged

    @staticmethod
    def add_rolling_features(df, columns):
        df_rolling = df.copy()
        for col in columns:
            if col not in df_rolling.columns: continue
            for window in config.ROLLING_WINDOWS:
                df_rolling[f'{col}_roll_{window}_mean'] = df_rolling[col].rolling(window=window).mean()
                df_rolling[f'{col}_roll_{window}_std'] = df_rolling[col].rolling(window=window).std()
        return df_rolling

    @staticmethod
    def encode_categories(df, col='layer'):
        if col in df.columns:
            df[f'{col}_code'] = df[col].astype('category').cat.codes
        return df

    @staticmethod
    def apply_normalization(df, columns):
        df_norm = df.copy()
        for col in columns:
            if col in df_norm.columns:
                min_val, max_val = df_norm[col].min(), df_norm[col].max()
                if max_val - min_val != 0:
                    df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
        return df_norm
