import numpy as np
import scipy.optimize as opt


def f_opt(x):
    x1, x2 = x
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def h_const(x):
    x1, x2 = x
    return (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2 - 0.25


def cop(f, h, x0, eps, a0, beta):
    hk = np.inf
    xk = x0
    ak = a0

    while abs(hk) >= eps:
        xkp1 = penalty_method(f, h, xk, ak)
        hk = h(xkp1)
        xk = xkp1
        ak = beta * ak

    return xk


def penalty_method(f, h, xk, ak):

    def phi(x, ak):
        return f(x) + 0.5 * ak * (h(x)) ** 2

    xkp1 = opt.minimize(
        phi, xk, method="nelder-mead", args=(ak)
    )  # result of minimization of phik
    return xkp1
