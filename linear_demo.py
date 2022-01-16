import torch
import torch.nn as nn
import numpy as np

dataset_numpy = np.loadtxt("data2.txt", delimiter=',')
dataset = torch.from_numpy(dataset_numpy[:, 0:2]).float()
y = torch.from_numpy(dataset_numpy[:, -1]).float()


# TODO: maybe we should change the defaulted type(dtype) to float in Pycharm.
# ATTENTION: when importing data from a numpy array, we should change the type to torch.float32 or errors may occur.

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        # to inherit the parent class that has a lot of built-in functions.
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        predict = self.linear(x)
        return predict


'''
When claiming a network, or a computing unit, the __init__ and __forward__ are quite essential steps.
We may got bewildered by 'linear', for they appeared twice both in __init__ and __forward__. 
But when we look into the internal codes, our puzzles may become a bit clearer.

in Line 14:
    self.linear = nn.Linear(2,1)
    -def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
    we use this to initialize a linear function to compute:
                y = w * x + b
    where:
                w: weights
                x: input
                b: bias value
    
in Line 17:
    we will call the objectified "linear" to return a value.
'''

model = Linear()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10)

for epoch in range(1000):
    for j in range(len(dataset)):
        pred = model(dataset[j])
        loss = criterion(pred, y[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for i in range(len(dataset)):
    pred = model(dataset[i])
    print(i, pred.item(), y[i].item())
