import numpy as np
import scipy.optimize as opt


def f_opt(x):
    x1, x2 = x
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def h_const(x):
    x1, x2 = x
    return (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2 - 0.25


def cop(f, h, x0, eps, lambda0, a0, beta):
    hk = np.inf
    xk = x0
    ak = a0
    lambdak = lambda0

    while abs(hk) >= eps:
        xkp1 = lagrange_multiplier(f, h, xk, lambdak, ak)
        hk = h(xkp1)
        xk = xkp1
        lambdak = lambdak + ak * hk
        ak = beta * ak

    return xk


def lagrange_multiplier(f, h, xk, lambdak, ak):
    def psi(x, lam, a):
        return f(x) + lam * h(x) + 0.5 * a * (h(x)) ** 2

    xkp1 = opt.minimize(
        psi, xk, method="nelder-mead", args=(lambdak, ak)
    )  # result of minimization of phik
    return xkp1
