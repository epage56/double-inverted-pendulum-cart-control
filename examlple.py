import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A, B, Q, R):
    """Solves the discrete-time LQR problem."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return -K

# === Physical parameters ===
l = 0.075 / 2      # COM distance (m)
m = 0.035          # Pendulum mass (kg)
M = 0.1            # Cart mass (kg)
g = 9.8            # Gravity (m/s^2)
dt = 0.02          # Time step (s)

# === Discrete-time linearized dynamics ===
den = 7*M + 4*m
A = np.array([
    [1, dt, 0, 0],
    [0, 1, -(3*m*g*dt)/den, 0],
    [0, 0, 1, dt],
    [0, 0, (3*g*(m+M)*dt)/(l*den), 1]
])
B = np.array([
    [0],
    [7*dt/den],
    [0],
    [-3*dt/(l*den)]
])

# === LQR setup ===
Q = np.diag([1.0, 0.0001, 1.0, 0.0001])
R = np.array([[0.0005]])
K = dlqr(A, B, Q, R)

# === Print controller gains ===
print("K =", K)
print("double c[] = {%.6f, %.6f, %.6f, %.6f};" % tuple(K[0]))

# === Motor model coefficients (from fit) ===
a_motor = 0.432
b_motor = -0.004
c_motor = -0.708  # static friction

# === Simulate system ===
N = 250
x = np.array([[0], [0], [0.4], [0]])  # initial state: x, dx, theta, dtheta
X, T, U = [], [], []

for _ in range(N):
    u = float(K @ x)                   # control force in Newtons
    v = x[1, 0]                        # cart velocity
    friction = c_motor * np.sign(v)   # friction force
    u_volt = ((1 - a_motor) * v + dt * u / (M + m) + friction) / b_motor
    X.append(x[0, 0])
    T.append(x[2, 0])
    U.append(u_volt)
    x = A @ x + B * u
    
# === Plot ===
t = np.linspace(0, N*dt, N)
plt.plot(t, X, label="cart position (m)")
plt.plot(t, T, label="pendulum angle (rad)")
plt.plot(t, U, label="control voltage (decavolts)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()