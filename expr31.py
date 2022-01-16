import torch
import torch.nn.functional as F


x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())
loss = F.mse_loss(o, torch.ones(1, 2))
loss.backward()
print(w.grad)