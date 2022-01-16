import numpy as np
import matplotlib.pyplot as plt
from main import fun

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
# print('x,y range', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
# print('X,Y maps',X.shape, Y.shape)
Z = fun([X,Y])

fig = plt.figure('fun')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()