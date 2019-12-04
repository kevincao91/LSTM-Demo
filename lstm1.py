import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


# RNN 模型
class RNN(nn.Module):
    def __init__(self, look_back):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.lstm = nn.LSTM(look_back, 6, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.out = nn.Linear(6, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1


# 数据集和目标值赋值，data_set为数据，look_back为以几行数据为特征维度数量
def create_data_set(data_set, look_back):
    data_x = []
    data_y = []
    for i in range(len(data_set) - look_back):
        data_x.append(data_set[i:i + look_back])
        data_y.append(data_set[i + look_back])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据


# 一、数据准备

# 参考长度
look_back = 3
'''
# sin
x = np.arange(1, 10, 0.1)
x = np.sin(x)
datas = x[:]
'''

# 本地数据 
df1 = pd.read_excel(r'simple1.xlsx')
# df1 = pd.read_excel(r'Stock_Open_Data.xlsx')
datas = df1.values
datas = datas[:]


print('read data ok')
# 归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用
max_value = np.max(datas)
min_value = np.min(datas)
scalar = max_value - min_value
datas = list(map(lambda x: (x-min_value) / scalar, datas))


# 以2为特征维度，得到数据集
dataX, dataY = create_data_set(datas, look_back)
train_size = int(len(dataX) * 0.7)
x_train = dataX[:train_size]  # 训练数据
y_train = dataY[:train_size]  # 训练数据目标值
x_train = x_train.reshape(-1, 1, look_back)  # 将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度

# 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
print('read data ok')

# 二、创建LSTM模型
rnn = RNN(look_back)

# 参数寻优，计算损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
loss_func = nn.MSELoss()
print('read data ok')

# 三、训练模型
for i in range(500):
    var_x = Variable(x_train).type(torch.FloatTensor)
    var_y = Variable(y_train).type(torch.FloatTensor)
    out = rnn(var_x)
    loss = loss_func(out, var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))

# 四、模型测试
# 准备测试数据
dataX1 = dataX.reshape(-1, 1, look_back)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor)

pred = rnn(var_dataX)
pred_data = pred.view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值  dataY为真实值

# 五、画图检验
plt.plot(pred_data, 'r', label='prediction')
plt.plot(train_size+1, pred_data[train_size+1], 'o', label='prediction')
plt.plot(dataY, 'b', label='real')
plt.legend(loc='best')
plt.show()
