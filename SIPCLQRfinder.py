import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from matplotlib.animation import FuncAnimation

# --- 1) SYSTEM PARAMETERS ---
M     = 0.1      # cart mass (kg)
m     = 0.035    # pendulum mass (kg)
l     = 0.075    # pendulum length (m)
g     = 9.81     # gravity (m/s^2)
b_cart = 0.5     # cart viscous damping (adjusted to realistic)
b_pend = 0.001   # pendulum viscous damping (N·m·s/rad)

# --- 2) NONLINEAR DYNAMICS (for simulation only) ---
def f_cont(y, u):
    x, th, x_dot, th_dot = y
    s, c = np.sin(th), np.cos(th)
    D = M + m * s**2
    x_dd  = (u + m*l*th_dot**2*s - m*g*s*c - b_cart*x_dot)/D
    th_dd = (-u*c - m*l*th_dot**2*s*c + (M+m)*g*s - b_cart*x_dot*c - b_pend*th_dot)/(l*D)
    return np.array([x_dot, th_dot, x_dd, th_dd])

# --- 3) LINEARIZATION AT EQUILIBRIUM y_eq = [0, 0, 0, 0], u_eq = 0 ---
# analytic A, B for inverted pendulum about upright θ=0:
den = M+m
A = np.array([
    [0,         0,            1,             0],
    [0,         0,            0,             1],
    [0, m*g/den, -b_cart/den,  0],
    [0, (den)*g/(l*den), -b_cart/(l*den), -b_pend/(l*den)]
])
B = np.array([[0],
              [0],
              [1/den],
              [ -1/(l*den) ]])

# --- 4) DISCRETIZE (zero‐order hold) ---
dt = 10.0/2000      # same as before: T=10, N=2000
sysd = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), dt, method='zoh')
Ad, Bd = sysd[0], sysd[1].flatten()

# --- 5) DISCRETE‐TIME LQR DESIGN ---
Qd = np.diag([10, 100, 1, 10])
Rd = np.array([[2.4]])
# Solve discrete-time ARE:
Pd = solve_discrete_are(Ad, Bd[:,None], Qd, Rd)
# Compute gain Kd = (BᵀP B + R)⁻¹ (BᵀP A)
Kd = np.linalg.inv(Bd[:,None].T @ Pd @ Bd[:,None] + Rd) @ (Bd[:,None].T @ Pd @ Ad)
Kd = Kd.flatten()
print("Discrete LQR gain Kd =", Kd)

# --- 6) DISCRETE‐TIME SIMULATION ---
T, N = 10.0, 2000
t = np.linspace(0, T, N)
y = np.zeros((4, N))
u = np.zeros(N)
y[:,0] = np.array([0.0, 0.3, 0.0, 0.0])  # small initial tilt

for k in range(N-1):
    # state error about upright
    e = y[:,k]
    # control law (with simple saturation)
    uk = -Kd @ e
    uk = np.clip(uk, -50, 50)
    u[k] = uk
    # discrete‐time update
    y[:,k+1] = Ad @ y[:,k] + Bd * uk

u[-1] = u[-2]

# --- 7) PLOT RESULTS ---
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
ax1.plot(t, y[0], label='x (m)');     ax1.set_ylabel('Cart pos');    ax1.grid(True)
ax2.plot(t, y[1], label='θ (rad)');   ax2.set_ylabel('Pend angle');  ax2.grid(True)
ax3.plot(t, u,   label='u (N)');      ax3.set_ylabel('Control');     ax3.set_xlabel('Time (s)'); ax3.grid(True)
plt.tight_layout()
plt.show()

# --- 8) ANIMATION (optional) ---
fig2, ax = plt.subplots(figsize=(6,4))
ax.set_xlim(-0.2,0.2); ax.set_ylim(-0.2,0.2); ax.set_aspect('equal')
cart_line, = ax.plot([],[], 's-', lw=2)
rod_line,  = ax.plot([],[], 'o-', lw=2)
time_text  = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    cart_line.set_data([],[]); rod_line.set_data([],[]); time_text.set_text('')
    return cart_line, rod_line, time_text

def update(frame):
    xi, thetai = y[0,frame], y[1,frame]
    px = xi + l*np.sin(thetai)
    py = l*np.cos(thetai)
    cart_line.set_data([xi-0.02, xi+0.02],[0,0])
    rod_line.set_data([xi, px],[0, py])
    time_text.set_text(f't={t[frame]:.2f}s')
    return cart_line, rod_line, time_text

ani = FuncAnimation(fig2, update, frames=N, init_func=init, interval=dt*1000, blit=True)
plt.show()