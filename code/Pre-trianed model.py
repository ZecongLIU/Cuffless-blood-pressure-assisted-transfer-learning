import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
file_path = 'ZONG.csv'  # 请替换为您的文件路径
df = pd.read_csv(file_path)

# 确保所有数据都是数值型
df = df.apply(pd.to_numeric, errors='coerce')

# 填充或删除缺失值
df.fillna(0, inplace=True)  # 用0填充缺失值，或者您可以选择删除这些行：df.dropna(inplace=True)
# 设定采样频率
fs = 1000  # 采样频率100Hz，根据实际情况调整
nyquist = 0.5 * fs
low = 1 / nyquist
high = 10 / nyquist
b, a = signal.butter(N=4, Wn=[low, high], btype='band')

# 应用带通滤波器
df_filtered = df.iloc[5000:10000, 1:41].apply(lambda x: signal.filtfilt(b, a, x))

# 处理目标值
targets = df.iloc[-2:, 1:41].T
# 转换为张量
data_tensor = torch.tensor(df_filtered.values.T, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

# 重塑数据为适合1D CNN的格式 (样本数, 特征数, 时间步长)
data_tensor = data_tensor.unsqueeze(1)  # 增加一个维度
# 确保样本数匹配
assert data_tensor.shape[0] == targets_tensor.shape[0], "样本数不匹配"
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data_tensor, targets_tensor, test_size=0.125, random_state=40)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1)

# 输出数据形状以确认
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# 定义1D CNN模型
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=X_train.shape[1], out_channels=8, kernel_size=2500)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_size = X_train.shape[2] - 2500 + 1  # 根据公式计算
        self.fc1 = nn.Linear(8 * conv_output_size, 50)
        self.fc2 = nn.Linear(50, y_train.shape[1])  # 假设y_train的每个样本有多个输出值

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = CNN1D()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# 初始化最佳验证损失为无穷大
best_val_loss = float('inf')
# 训练模型
for epoch in range(20):  # 迭代次数
    # 训练阶段
    model.train()
    total_train_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # 验证阶段
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)

    # 打印训练和验证损失
    print(f"Epoch {epoch+1}, Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
    # 检查是否有更好的验证损失
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # 保存最优模型
        torch.save(model.state_dict(), 'best_model.pth')



# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
# 测试模型并收集预测输出及真实值
model.eval()
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        # 收集预测值和真实值
        predictions.extend(output.tolist())
        actuals.extend(target.tolist())

# 打印测试损失、预测结果和真实值
print(f"Test loss: {test_loss / len(test_loader)}")
print("Predictions on Test Data:")
print(predictions)
print("Actual Values:")
print(actuals)
