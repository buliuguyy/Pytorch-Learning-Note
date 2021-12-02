import numpy as np
import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn

def load_array(data_arrays, batch_size, is_shuffle=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_shuffle)


if __name__ == "__main__":
    true_w = torch.Tensor([2.4, -3.2])  # 必须转成tensor的形式
    true_b = 4.2
    num_examples = 1000
    batch_size = 10
    features, labels = d2l.synthetic_data(true_w, true_b, num_examples)
    data_iter = load_array((features, labels), batch_size)

    # 通过next函数转为Python中的迭代器
    next(iter(data_iter))

    # 使用架构的预定义好的层
    net = nn.Sequential(nn.Linear(2,1))

    # 初始化模型参数
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)

    # loss选用均方误差——L2范数
    loss = nn.MSELoss()

    # 实例化SGD实例
    trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

    # 开始训练
    num_epochs = 3
    for epoch in range(0, num_epochs, 1):
        for X,y in data_iter:
            l = loss(net(X),y)
            trainer.zero_grad()  # 梯度清0，pytorch不清0，如果不手动清就一直累加了
            l.backward()         # y是向量时，pytorch已经做好了变量sum合成新变量的过程，不需要自己写
            trainer.step()       # 进行单步的优化，参数更新
        l = loss(net(features), labels)
        print(f'epoch {epoch+1}, loss {l:f}')