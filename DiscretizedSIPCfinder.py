import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# === System parameters ===
M = 0.1     # kg
m = 0.035   # kg
l = 0.075   # m
g = 9.81    # m/s²
dt = 0.005  # s

I = m * l**2
D = I * (M + m) + M * m * l**2

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m * g * l) / D, 0],
    [0, 0, 0, 1],
    [0, 0, (g * l * (M + m)) / D, 0]
])
B = np.array([
    [0],
    [(I + m * l**2) / D],
    [0],
    [-m * l / D]
])
Ad = np.eye(4) + A * dt
Bd = B * dt

# === LQR weights ===
Q = np.diag([1.0, 0.0001, 1.0, 0.0001])
R = np.array([[0.1]])

# === Solve DARE ===
P = solve_discrete_are(Ad, Bd, Q, R)
K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)

print(K)

# === Simulate response ===
N = 1000  # number of steps
x = np.zeros((4, N))
u = np.zeros(N)

# Initial state: small angle offset, rest is zero
x[:, 0] = [0.0, 0.0, np.deg2rad(5.0), 0.0]  # 10 degrees

for k in range(N-1):
    u[k] = -K @ x[:, k]
    x[:, k+1] = Ad @ x[:, k] + Bd.flatten() * u[k]

# === Plot ===
time = np.arange(N) * dt
fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

labels = ["x (m)", "dx (m/s)", "theta (deg)", "dtheta (deg/s)", "F (N)"]
data = [x[0], x[1], np.rad2deg(x[2]), np.rad2deg(x[3]), u]

for i in range(5):
    axs[i].plot(time, data[i])
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)

axs[-1].set_xlabel("Time (s)")
plt.suptitle("Closed-loop LQR Response")
plt.tight_layout()
plt.show()