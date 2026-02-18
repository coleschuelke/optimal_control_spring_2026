import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def f_opt(x):
    x1, x2 = x
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def h_const(x):
    x1, x2 = x
    return (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2 - 0.25


def cop(f, h, x0, eps, lambda0, a0, beta):
    x_hist = []

    hk = np.inf
    xk = x0
    ak = a0
    lambdak = lambda0

    x_hist.append(x0)

    while abs(hk) >= eps:
        xkp1 = lagrange_multiplier(f, h, xk, lambdak, ak)
        hk = h(xkp1)
        xk = xkp1
        lambdak = lambdak + ak * hk
        ak = beta * ak
        x_hist.append(xk)

    return (xk, np.array(x_hist))


def lagrange_multiplier(f, h, xk, lambdak, ak):
    def psi(x, lam, a):
        return f(x) + lam * h(x) + 0.5 * a * (h(x)) ** 2

    xkp1 = opt.minimize(
        psi, xk, method="nelder-mead", args=(lambdak, ak)
    )  # result of minimization of phik
    return xkp1.x


# Numerical solution
a0 = 0.1
beta = 6
eps = 1e-4
lambda0 = 10
x0 = np.array([1, 1]).T

x_tilde_star, x_hist = cop(f_opt, h_const, x0, eps, lambda0, a0, beta)
print(f"The numerical solution is: \n {x_tilde_star}")
print(f"The solution converged after {x_hist.shape[0]-1} steps. ")
print(f"The final value of the constraint function is: {h_const(x_tilde_star)}")


# Plotting
def f_plot(x1, x2):
    return f_opt(np.array([x1, x2]).T)


def h_plot(x1, x2):
    return h_const(np.array([x1, x2]).T)


f_vecd = np.vectorize(f_plot)
h_vecd = np.vectorize(h_plot)

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(x1, x2)
z = f_vecd(x1, x2)
z2 = h_vecd(x1, x2)

fig, ax = plt.subplots()
f_contours = ax.contour(x1, x2, z)
h_contours = ax.contour(x1, x2, z2)
ax.plot(x_hist[:, 0], x_hist[:, 1], marker="x", linestyle="dashed", color="b")
ax.clabel(f_contours, inline=True, fontsize=10, fmt="%.1f")
ax.clabel(h_contours, inline=True, fontsize=10, fmt="%.1f")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Minimization trajectory")

plt.show()
