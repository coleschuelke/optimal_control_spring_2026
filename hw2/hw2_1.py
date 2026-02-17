import numpy as np


def uop(Q: np.ndarray, b: np.ndarray, x0: np.ndarray, eps: float):

    residual = np.inf
    xk = x0

    while residual > eps:
        xkp1 = step(Q, b, xk)
        residual = abs(xkp1 - xk)
        xk = xkp1

    return xkp1


def step(Q: np.ndarray, b: np.ndarray, xk: np.ndarray):
    ak = 0  # TODO: Find analytical expression for ak

    xkp1 = xk - ak * (Q @ xk - b)

    return xkp1
