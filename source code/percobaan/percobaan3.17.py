import numpy as np
import scipy.optimize as optimization

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([100, 90, 60, 30, 10, 1])


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


popt, pcov = optimization.curve_fit(func, x, y, p0=[100, 0.5, 1])

print("Best fit a b c: ", popt)
print("Best fit covariance: ", pcov)