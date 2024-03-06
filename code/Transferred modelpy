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
file_path = 'qianyixuan.csv'  # 请替换为您的文件路径
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
df_filtered = df.iloc[:5000, 1:7].apply(lambda x: signal.filtfilt(b, a, x))

# 处理目标值
targets = df.iloc[-2:, 1:7].T
# 转换为张量
data_tensor = torch.tensor(df_filtered.values.T, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

# 重塑数据为适合1D CNN的格式 (样本数, 特征数, 时间步长)
data_tensor = data_tensor.unsqueeze(1)  # 增加一个维度
# 确保样本数匹配
assert data_tensor.shape[0] == targets_tensor.shape[0], "样本数不匹配"
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data_tensor, targets_tensor, test_size=5/6, random_state=43)

# 创建 DataLoader
new_train_dataset = TensorDataset(X_train, y_train)
new_train_loader = DataLoader(new_train_dataset, batch_size=1, shuffle=True)

new_test_dataset = TensorDataset(X_test, y_test)
new_test_loader = DataLoader(new_test_dataset, batch_size=1)

# 输出数据形状以确认
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        # 调整卷积层和全连接层以匹配预训练模型
        self.conv1 = nn.Conv1d(in_channels=X_train.shape[1], out_channels=8, kernel_size=2500)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加dropout层，假设dropout率为0.5
        self.flatten = nn.Flatten()
        conv_output_size = X_train.shape[2] - 2500 + 1
        self.fc1 = nn.Linear(8 * conv_output_size, 50)
        self.fc2 = nn.Linear(50, y_train.shape[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 在激活函数之后应用dropout
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 加载预训练模型
pretrained_model = CNN1D()

# 加载预训练模型权重
pretrained_model.load_state_dict(torch.load('best_model.pth'))

# 冻结预训练模型的部分层
for param in pretrained_model.parameters():
    param.requires_grad = False

# 重新激活最后几层的学习
for param in pretrained_model.fc1.parameters():
    param.requires_grad = True
for param in pretrained_model.fc2.parameters():
    param.requires_grad = True


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.000005, weight_decay=1e-5)

# 在训练模型前定义早停参数
patience = 5  # 阈值：连续多少个epochs没有改善
no_improve_epochs = 0  # 连续没有改善的epochs数
# 初始化最佳验证损失为无穷大
best_val_loss = float('inf')

# 训练模型
for epoch in range(500):  # 迭代次数
    # 训练阶段
    pretrained_model.train()
    total_train_loss = 0
    l1_lambda = 0.0001  # 设置L1正则化强度
    l1_norm = sum(p.abs().sum() for p in pretrained_model.parameters())
    for data, target in new_train_loader:
        optimizer.zero_grad()
        output = pretrained_model(data)
        loss = criterion(output, target)
        loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(new_train_loader)

    # 验证阶段
    pretrained_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, target in new_test_loader:
            output = pretrained_model(data)
            loss = criterion(output, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(new_test_loader)

    # 打印训练和验证损失
    print(f"Epoch {epoch+1}, Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
    # 检查是否有更好的验证损失
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        # 保存最优模型
        torch.save(pretrained_model.state_dict(), 'new_best_model.pth')
    else:
        no_improve_epochs += 1
    
    # 检查是否应该执行早停
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break



# 加载最佳模型
pretrained_model.load_state_dict(torch.load('new_best_model.pth'))
# 测试模型并收集预测输出及真实值
pretrained_model.eval()
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for data, target in new_test_loader:
        output = pretrained_model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        # 收集预测值和真实值
        predictions.extend(output.tolist())
        actuals.extend(target.tolist())

# 打印测试损失、预测结果和真实值
print(f"Test loss: {test_loss / len(new_test_loader)}")
print("Predictions on Test Data:")
print(predictions)
print("Actual Values:")
print(actuals)
