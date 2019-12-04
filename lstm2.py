import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import random


# RNN 模型
class RNN(nn.Module):
    def __init__(self, look_back):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.lstm = nn.LSTM(look_back, 8, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.out = nn.Linear(8, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为2

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
    for data in data_set:
        for i in range(len(data) - look_back):
            data_x.append(data[i:i + look_back])
            data_y.append(data[i + look_back])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据


# 数据集和目标值赋值，data_set为数据，look_back为以几行数据为特征维度数量
def create_test_data_set(data_set, look_back):
    data_x = []
    data_y = []
    for i in range(len(data_set) - look_back):
        data_x.append(data_set[i:i + look_back])
        data_y.append(data_set[i + look_back])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据


# 一、数据准备
# 参考长度
look_back = 5

# 本地数据

with open(r'path_data.txt', 'r') as f:
    lines = f.readlines()

all_path_data = []
for line in lines:
    if '=' not in line:
        if ',' not in line:
            one_path_data = []
        else:

            one_path_data.append(np.asarray(list(map(int, line.split(',')[:2]))))
    else:
        all_path_data.append(one_path_data)

# print(all_path_data)


print('read data ok')
# 归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用

max_min_list = []
for datas in all_path_data:
    max_min_list.append(np.max(datas))
    max_min_list.append(np.min(datas))

max_value = np.max(max_min_list)
min_value = np.min(max_min_list)
scalar = max_value - min_value

nor_all_path_data = []
for datas in all_path_data:
    nor_all_path_data.append(list(map(lambda x: (x - min_value) / scalar, datas)))


datas = nor_all_path_data
# 以look_back为特征维度，得到数据集
dataX, dataY = create_data_set(datas, look_back)
dataX = dataX.transpose((0, 2, 1))
print('data numble: ', len(dataX))

train_size = int(len(dataX) * 0.7)
x_train = dataX[:train_size]  # 训练数据
y_train = dataY[:train_size]  # 训练数据目标值
x_train = x_train.reshape(-1, 2, look_back)  # 将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, 2, 1)  # 将目标值调整成pytorch中lstm算法的输出维度

# 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
print('train data ok')

# 二、创建LSTM模型
rnn = RNN(look_back)
rnn = rnn.cuda()

# 参数寻优，计算损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
loss_func = nn.MSELoss()
print('train net ok')

# 三、训练模型
for i in range(500):
    var_x = Variable(x_train).type(torch.FloatTensor).cuda()
    var_y = Variable(y_train).type(torch.FloatTensor).cuda()
    out = rnn(var_x)
    loss = loss_func(out, var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))


# 四、模型测试
# 准备测试数据
dlen = len(datas)
#  1/4
nn = random.randint(0, dlen)
plt.subplot(221)
plt.title(nn, fontsize='large', fontweight='bold')

# 本地数据
datas_test = datas[nn]

# 以look_back为特征维度，得到数据集
dataX, dataY = create_test_data_set(datas_test, look_back)
dataX = dataX.transpose((0, 2, 1))


dataX1 = dataX.reshape(-1, 2, look_back)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor).cuda()


rnn.eval()
pred = rnn(var_dataX)
pred_data = pred.view(-1, 2).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值  dataY为真实值

# 五、画图检验

plt.plot(pred_data[:, 0], pred_data[:, 1], 'r', label='prediction')
# plt.plot(pred_data[train_size+1, 0], pred_data[train_size+1, 1], 'o', label='start point')
plt.plot(dataY[:, 0], dataY[:, 1], 'b', label='real')
plt.legend(loc='best')
# x 轴逆序
# plt.gca().invert_xaxis()
# y 轴逆序
plt.gca().invert_yaxis()
plt.ion()
plt.show()

#  2/4
nn = random.randint(0, dlen)
plt.subplot(222)
plt.title(nn, fontsize='large', fontweight='bold')

# 本地数据
datas_test = datas[nn]

# 以look_back为特征维度，得到数据集
dataX, dataY = create_test_data_set(datas_test, look_back)
dataX = dataX.transpose((0, 2, 1))


dataX1 = dataX.reshape(-1, 2, look_back)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor).cuda()


rnn.eval()
pred = rnn(var_dataX)
pred_data = pred.view(-1, 2).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值  dataY为真实值

# 五、画图检验

plt.plot(pred_data[:, 0], pred_data[:, 1], 'r', label='prediction')
# plt.plot(pred_data[train_size+1, 0], pred_data[train_size+1, 1], 'o', label='start point')
plt.plot(dataY[:, 0], dataY[:, 1], 'b', label='real')
plt.legend(loc='best')
# x 轴逆序
# plt.gca().invert_xaxis()
# y 轴逆序
plt.gca().invert_yaxis()

plt.show()

#  3/4
nn = random.randint(0, dlen)
plt.subplot(223)
plt.title(nn, fontsize='large', fontweight='bold')

# 本地数据
datas_test = datas[nn]

# 以look_back为特征维度，得到数据集
dataX, dataY = create_test_data_set(datas_test, look_back)
dataX = dataX.transpose((0, 2, 1))


dataX1 = dataX.reshape(-1, 2, look_back)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor).cuda()


rnn.eval()
pred = rnn(var_dataX)
pred_data = pred.view(-1, 2).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值  dataY为真实值

# 五、画图检验
plt.plot(pred_data[:, 0], pred_data[:, 1], 'r', label='prediction')
# plt.plot(pred_data[train_size+1, 0], pred_data[train_size+1, 1], 'o', label='start point')
plt.plot(dataY[:, 0], dataY[:, 1], 'b', label='real')
plt.legend(loc='best')
# x 轴逆序
# plt.gca().invert_xaxis()
# y 轴逆序
plt.gca().invert_yaxis()

plt.show()


#  4/4
nn = random.randint(0, dlen)
plt.subplot(224)
plt.title(nn, fontsize='large', fontweight='bold')

# 本地数据
datas_test = datas[nn]

# 以look_back为特征维度，得到数据集
dataX, dataY = create_test_data_set(datas_test, look_back)
dataX = dataX.transpose((0, 2, 1))


dataX1 = dataX.reshape(-1, 2, look_back)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor).cuda()


rnn.eval()
pred = rnn(var_dataX)
pred_data = pred.view(-1, 2).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值  dataY为真实值

# 五、画图检验
plt.plot(pred_data[:, 0], pred_data[:, 1], 'r', label='prediction')
# plt.plot(pred_data[train_size+1, 0], pred_data[train_size+1, 1], 'o', label='start point')
plt.plot(dataY[:, 0], dataY[:, 1], 'b', label='real')
plt.legend(loc='best')
# x 轴逆序
# plt.gca().invert_xaxis()
# y 轴逆序
plt.gca().invert_yaxis()
plt.ioff()
plt.show()