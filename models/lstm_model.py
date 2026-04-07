import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMPredictor(nn.Module):
    """
    基于 PyTorch 的 LSTM 回归模型定义
    支持多层堆叠、双向 LSTM 及 Dropout。
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, bidirectional=True, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接输出层
        # 若为双向，隐藏层维度需乘以 2
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)
        
        # 我们取最后一个时间步的输出作为预测结果
        last_time_step = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step)
        return out

class LSTMTrainer:
    """
    LSTM 训练管理器：负责数据转换、循环训练与模型保存。
    """
    def __init__(self, model=None, lr=0.001, device='cpu'):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def prepare_data(self, X, y, seq_len=7):
        """
        将 2D 数组转换为 3D 时序张量 (N, Seq_Len, Features)
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len])
        
        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32).view(-1, 1)
        return X_tensor, y_tensor

    def train(self, train_loader, epochs=100, val_loader=None):
        """
        增强版训练循环：包含早停逻辑雏形和更好的日志输出
        """
        print(f"[LSTM] 正在以深度学习模式训练 (Epochs: {epochs})...")
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            if (epoch + 1) % 5 == 0:
                val_info = ""
                if val_loader:
                    val_loss = self._validate(val_loader)
                    val_info = f" | Val Loss: {val_loss:.6f}"
                    if val_loss < best_loss:
                        best_loss = val_loss
                        # 自动保存当前最优模型
                        torch.save(self.model.state_dict(), "temp_best_lstm.pth")
                print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.6f}{val_info}")

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                val_loss += self.criterion(output, batch_y).item()
        return val_loss / len(val_loader)

    def predict(self, test_x):
        self.model.eval()
        with torch.no_grad():
            test_x = test_x.to(self.device)
            return self.model(test_x).cpu().numpy()

    def save_model(self, path):
        """
        保存模型权重
        """
        torch.save(self.model.state_dict(), path)
        print(f"[LSTM] 模型已保存至: {path}")

    def load_model(self, path):
        """
        加载模型权重
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"[LSTM] 已从 {path} 加载模型权重。")
