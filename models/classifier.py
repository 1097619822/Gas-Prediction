import torch
import torch.nn as nn
import torch.optim as optim

class LSTMClassifier(nn.Module):
    """
    用于积液等级分类的 LSTM 模型。
    输出层：多分类 (Softmax)。
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # 双向 LSTM 维度翻倍
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] # 取最后一个时间步
        logits = self.fc(last_out)
        return logits

class ClassificationTrainer:
    """
    分类任务训练器。
    """
    def __init__(self, model, lr=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        # 分类任务使用交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, train_loader, val_loader, epochs=30):
        print(f"[Diagnosis] 开始训练分类模型...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(bx)
                loss = self.criterion(outputs, by.long().squeeze())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")
