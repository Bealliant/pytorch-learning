import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

dataset_numpy = np.loadtxt("data.txt", delimiter=",", dtype=float)
dataset_numpy = np.insert(dataset_numpy, 0, values=np.ones((1, 118)), axis=1)
dataset = torch.from_numpy(dataset_numpy[:, 0:3]).reshape(118, 3, 1).float()


def logistic_predict(weight, k):
    z = torch.mm(weight[0], dataset[k])
    a = torch.sigmoid(z)
    z2 = torch.mm(weight[1], a)
    h = torch.sigmoid(z2)
    J = h.sum()/3
    return J


def logistic_mse_loss(weight, k):
    return (logistic_predict(weight, k)-y[k])**2


y = torch.from_numpy(dataset_numpy[:, -1])
w = torch.rand(2, 3, 3)
w.requires_grad = True
optimizer = torch.optim.Adam([w], lr=4*1e-2)

for epoch in range(50):
    i = 0
    for k in range(len(dataset)):
        loss = logistic_mse_loss(w, k)
        if epoch == 49:
            print(k, loss, y[k], logistic_predict(w,k))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(w)