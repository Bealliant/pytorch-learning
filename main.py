import numpy as np
import matplotlib.pyplot as plt
import torch

def fun(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


x = torch.Tensor(2)
x.requires_grad = True
optimizer = torch.optim.Adam([x],lr=1e-3)
for step in range(20000):

    pred = fun(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step% 2000 == 0:
        print('step{}: x = {}, f(x) = {}'.format(step,x.tolist(),pred.item()))