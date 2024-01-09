import matplotlib.pyplot as plt
import numpy as np
import pylab as pb

from matplotlib import cm
from matplotlib.ticker import LinearLocator

def function_mse(w, b, y: float):
    return (1/(1 + np.exp(-(w + b))) - y)**2

def plot_function():
    x = np.arange(-5, 5, 0.001)

    y = np.arange(-5, 5, 0.001)
    X, Y = np.meshgrid(x, y)
    Z = function_mse(X, Y, 0.3)
    image = pb.imshow(Z, cmap=cm.coolwarm)
    pb.title("MSE with y = 1")
    pb.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


plot_function()
