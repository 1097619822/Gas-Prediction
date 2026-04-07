from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class BaselineModels:
    """
    基准模型类：封装 ARIMA, SVM, RF 等传统模型。
    """
    
    @staticmethod
    def train_svm(x_train, y_train):
        """
        训练支持向量回归模型 (SVR)
        """
        print("[Models] 正在训练 SVM (SVR) 模型...")
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def train_rf(x_train, y_train):
        """
        训练随机森林回归模型 (RF)
        """
        print("[Models] 正在训练 Random Forest 模型...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def run_arima(series, order=(5,1,0)):
        """
        执行 ARIMA 统计模型 (无需显式 fit，通常用于滚动预测)
        :param series: 产量序列
        :param order: (p, d, q) 参数
        """
        print(f"[Models] 正在运行 ARIMA{order} 模型...")
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
