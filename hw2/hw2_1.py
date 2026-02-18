import numpy as np
import matplotlib.pyplot as plt


# Numerical method
def g(Q: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return Q @ x.T - b


def step(Q: np.ndarray, b: np.ndarray, xk: np.ndarray):
    ak = (g(Q, b, xk).T @ Q @ xk - g(Q, b, xk).T @ b) / (
        g(Q, b, xk).T @ Q @ g(Q, b, xk)
    )
    xkp1 = xk - ak * g(Q, b, xk)

    return xkp1


def uop(Q: np.ndarray, b: np.ndarray, x0: np.ndarray, eps: float):
    x_hist = []

    residual = np.array([np.inf, np.inf]).T
    xk = x0
    x_hist.append(x0)

    while np.linalg.norm(residual) > eps:
        xkp1 = step(Q, b, xk)
        residual = abs(xkp1 - xk)
        xk = xkp1
        x_hist.append(xk)

    return (xkp1, np.array(x_hist))


# Define system Matrices
Q = np.array([[16, 3], [3, 4]])
b = np.array([2, 3]).T

# Analytical Solution
x_star = np.linalg.solve(Q.T, b)

print(f"The analytical solution is: \n {x_star}")

# Numerical solution
x01 = np.array([15, -4]).T
x02 = np.array([-12, 9]).T
x03 = np.array([-5, 13]).T
eps = 1e-3
x_tilde_star1, hist1 = uop(Q, b, x01, eps)
x_tilde_star2, hist2 = uop(Q, b, x02, eps)
x_tilde_star3, hist3 = uop(Q, b, x03, eps)
print("The numerical solutions are:")
print(f"{x_tilde_star1} after {hist1.shape[0]-1} iterations.")
print(f"{x_tilde_star2} after {hist2.shape[0]-1} iterations.")
print(f"{x_tilde_star3} after {hist3.shape[0]-1} iterations.")


# Plotting
err1 = hist1 - x_star
err2 = hist2 - x_star
err3 = hist3 - x_star

# Error vs iteration
fig, ax = plt.subplots()
ax.plot(np.linalg.norm(err1, axis=1), label=f"X0: {x01}")
ax.plot(np.linalg.norm(err2, axis=1), label=f"X0: {x02}")
ax.plot(np.linalg.norm(err3, axis=1), label=f"X0: {x03}")
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.set_title("Error vs Step")
ax.legend()


# Sequence of points on a contour plot
def f_plot(x1, x2):
    x = np.array([x1, x2]).T
    return 0.5 * x.T @ Q @ x - x.T @ b


x1 = np.linspace(-16, 16, 400)
x2 = np.linspace(-16, 16, 400)
x1, x2 = np.meshgrid(x1, x2)
f_vecd = np.vectorize(f_plot)
z = f_vecd(x1, x2)

fig, ax = plt.subplots()
contours = ax.contour(x1, x2, z)
ax.plot(hist1[:, 0], hist1[:, 1], marker="x", linestyle="dashed", color="b")
ax.plot(hist2[:, 0], hist2[:, 1], marker="x", linestyle="dashed", color="g")
ax.plot(hist3[:, 0], hist3[:, 1], marker="x", linestyle="dashed", color="m")
ax.scatter(hist1[0, 0], hist1[0, 1], marker="o", color="b")
ax.scatter(hist2[0, 0], hist2[0, 1], marker="o", color="g")
ax.scatter(hist3[0, 0], hist3[0, 1], marker="o", color="m")
ax.clabel(contours, inline=True, fontsize=10, fmt="%.1f")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Minimization trajectory for each initial condition")


plt.show()
