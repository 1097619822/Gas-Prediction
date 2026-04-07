import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class DataCleaner:
    """
    清洗器：负责 3-sigma、插值、格式统一。
    """
    @staticmethod
    def remove_outliers_3sigma(df, columns):
        df_cleaned = df.copy()
        for col in columns:
            if col not in df_cleaned.columns: continue
            mean, std = df_cleaned[col].mean(), df_cleaned[col].std()
            lower, upper = mean - config.SIGMA_THRESHOLD * std, mean + config.SIGMA_THRESHOLD * std
            df_cleaned.loc[(df_cleaned[col] < lower) | (df_cleaned[col] > upper), col] = np.nan
        return df_cleaned

    @staticmethod
    def handle_missing_values(df, columns, method=config.INTERPOLATE_METHOD):
        df_filled = df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].interpolate(method=method, limit_direction='both')
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        return df_filled

    @staticmethod
    def unify_formats(df):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
