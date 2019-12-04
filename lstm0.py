import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat


def series_gen(N):
    x = torch.arange(1, N, 0.01)
    return torch.sin(x)


def train_data_gen(seq, k):
    data = list()
    L = len(seq)
    for i in range(L - k - 1):
        in_data = seq[i:i + k]
        out_data = seq[i + 1:i + k + 1]
        data.append((in_data, out_data))
    return data


def to_variable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


if __name__ == '__main__':
    # print('yes')

    y = series_gen(10)
    train_data = train_data_gen(y.numpy(), 10)

    model = LSTMpred(1, 6)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        print(epoch)
        loss_sum = 0
        for inputs, outs in train_data[:700]:
            inputs = to_variable(inputs)
            outs = to_variable(outs)
            # outs = torch.from_numpy(np.array([outs]))

            optimizer.zero_grad()

            model.hidden = model.init_hidden()

            model_out = model(inputs)
            model_out = model_out.squeeze()

            loss = loss_function(model_out, outs)
            loss_sum += loss.detach().numpy()
            loss.backward()
            optimizer.step()
        print(loss_sum/700)

    pred_val = []
    model.eval()
    for seq, _ in train_data[:]:
        seq = to_variable(seq)
        pred_val.append(model(seq)[-1].data.numpy()[0])

    fig = plt.figure()
    plt.plot(y.numpy())
    plt.plot(pred_val)
    plt.show()
