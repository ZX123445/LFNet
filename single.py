import torch
import torch.nn as nn
import torch.optim as optim


# 定义双层感知机模型
class DoubleLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DoubleLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    # 实例化模型


input_size = 784  # 假设输入是28x28的图像，展平后为784个特征
hidden_size = 128  # 隐藏层大小
output_size = 10  # 假设有10个类别（例如MNIST数据集）
model = DoubleLayerPerceptron(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 假设你有一个输入数据x和对应的标签y
# x = torch.randn(batch_size, input_size)  # 随机生成的输入数据
# y = torch.randint(0, output_size, (batch_size,))  # 随机生成的标签

# 训练过程（此处省略了数据加载和循环）
# for epoch in range(num_epochs):
#     # ... (数据加载、前向传播、计算损失、反向传播、更新权重)

# 假设你已经有了x和y，进行前向传播和损失计算
# outputs = model(x)
# loss = criterion(outputs, y)

# 如果你想要保存模型，可以使用torch.save
# torch.save(model.state_dict(), 'model.pth')

# 加载保存的模型
# model.load_state_dict(torch.load('model.pth'))