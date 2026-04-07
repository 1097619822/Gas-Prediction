import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# 确保路径正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm_model import LSTMPredictor, LSTMTrainer

class PSOOptimizer:
    """
    PSO 优化器：用于自动搜索 LSTM 的最佳超参数。
    搜索空间：hidden_size, num_layers, learning_rate
    """
    def __init__(self, x_train, y_train, x_val, y_val, n_particles=5, n_iterations=3):
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # 定义搜索空间 (Min, Max)
        self.bounds = [
            (32, 256),    # hidden_size (int)
            (1, 4),       # num_layers (int)
            (1e-4, 1e-2)  # learning_rate (float)
        ]
        
        # 初始化粒子
        self.particles_pos = []
        for _ in range(n_particles):
            pos = [
                np.random.randint(self.bounds[0][0], self.bounds[0][1]),
                np.random.randint(self.bounds[1][0], self.bounds[1][1]),
                np.random.uniform(self.bounds[2][0], self.bounds[2][1])
            ]
            self.particles_pos.append(np.array(pos))
            
        self.particles_vel = [np.random.uniform(-1, 1, 3) for _ in range(n_particles)]
        self.p_best_pos = list(self.particles_pos)
        self.p_best_val = [float('inf')] * n_particles
        self.g_best_pos = None
        self.g_best_val = float('inf')

    def fitness_function(self, params):
        """
        适应度函数：使用给定的参数训练一个简易 LSTM，返回验证集 RMSE。
        """
        h_size, n_layers, lr = int(params[0]), int(params[1]), float(params[2])
        input_size = self.x_train.shape[2]
        
        # 构建临时模型
        model = LSTMPredictor(input_size=input_size, hidden_size=h_size, num_layers=n_layers)
        trainer = LSTMTrainer(model, lr=lr)
        
        # 为了搜索效率，这里只训练少量 Epoch
        train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=32, shuffle=False)
        val_loader = DataLoader(TensorDataset(self.x_val, self.y_val), batch_size=32, shuffle=False)
        
        trainer.train(train_loader, epochs=5) # 快速评估
        
        # 计算验证集 Loss
        val_loss = trainer._validate(val_loader)
        return val_loss

    def optimize(self):
        """
        PSO 核心迭代过程
        """
        print(f"[PSO] 开始优化 LSTM 超参数 (粒子数: {self.n_particles}, 迭代次数: {self.n_iterations})...")
        
        w, c1, c2 = 0.5, 1.5, 1.5 # PSO 标准参数
        
        for i in range(self.n_iterations):
            for p in range(self.n_particles):
                fitness = self.fitness_function(self.particles_pos[p])
                
                # 更新个体最优
                if fitness < self.p_best_val[p]:
                    self.p_best_val[p] = fitness
                    self.p_best_pos[p] = np.copy(self.particles_pos[p])
                
                # 更新全局最优
                if fitness < self.g_best_val:
                    self.g_best_val = fitness
                    self.g_best_pos = np.copy(self.particles_pos[p])
                    
            # 更新速度与位置
            for p in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.particles_vel[p] = (w * self.particles_vel[p] + 
                                         c1 * r1 * (self.p_best_pos[p] - self.particles_pos[p]) + 
                                         c2 * r2 * (self.g_best_pos - self.particles_pos[p]))
                
                self.particles_pos[p] = self.particles_pos[p] + self.particles_vel[p]
                
                # 边界约束
                for b in range(len(self.bounds)):
                    self.particles_pos[p][b] = np.clip(self.particles_pos[p][b], self.bounds[b][0], self.bounds[b][1])

            print(f"Iteration {i+1}/{self.n_iterations} 完成, 当前全局最优 Loss: {self.g_best_val:.6f}")

        print(f"\n[PSO] 优化完成！最佳参数: Hidden Size={int(self.g_best_pos[0])}, Layers={int(self.g_best_pos[1])}, LR={self.g_best_pos[2]:.6f}")
        return self.g_best_pos
