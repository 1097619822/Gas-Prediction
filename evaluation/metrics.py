import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    """
    评价模块：统一计算产量预测的各项指标。
    """
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """
        一次性计算 RMSE, MAE, MAPE, R2
        :param y_true: 真实值向量
        :param y_pred: 预测值向量
        :return: dict
        """
        # 确保输入是 numpy 数组
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # 过滤掉 0 值，防止 MAPE 计算溢出
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        results = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mape,
            'R2': r2_score(y_true, y_pred)
        }
        return results

    @staticmethod
    def print_results(metrics_dict, model_name="Model"):
        """
        控制台漂亮打印结果
        """
        print(f"\n[{model_name} 评估结果]")
        for k, v in metrics_dict.items():
            print(f" - {k:8s}: {v:.4f}")
