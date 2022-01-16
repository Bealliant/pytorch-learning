import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dataset_numpy = np.loadtxt("data.txt",delimiter=',')
dataset = torch.from_numpy(dataset_numpy[:, 0:2]).float()
dataset.requires_grad = True
temp_y = torch.from_numpy(dataset_numpy[:, -1]).float()
y = temp_y.reshape(len(dataset),1)

class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid, self).__init__()
        self.linear = nn.Linear(2,1)
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = sigmoid()
criterion = nn.BCELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.03)

for epoch in range(10000):
    print('\r' + '▇' * (epoch//100) + str(epoch/100) + '%', end='')
    y_pred = model(dataset)
    loss = criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for j in range(len(dataset)):
    pred = model(dataset[j])
    print(str(j)+ str(pred) + str(y[j]))

