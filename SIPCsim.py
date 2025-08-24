import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg      import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ─── toggle controller ─────────────────────────────────────────────────────────
use_lqr = False

# ─── params ─────────────────────────────────────────────────────────────────────
M, m, l, g = 0.5, 0.2, 0.3, 9.81
b_cart, b_pend = 0.2, 0.05

# ─── linearize & compute K ──────────────────────────────────────────────────────
den = (M + m)*m*l**2 + M*m**2*l**2
A = np.array([[0,0,1,0],
              [0,0,0,1],
              [0, -(m**2*g*l**2)/den, -b_cart*(m*l**2)/den,  b_pend*m*l/den],
              [0,  (M+m)*m*g*l/den,     b_cart*m*l/den,   -b_pend*(M+m)/den]])
B = np.array([[0],[0],[m*l**2/den],[-m*l*(M+m)/den]])
P = solve_continuous_are(A, B, np.diag([10,100,1,1]), np.eye(1))
K = np.linalg.inv(np.eye(1)) @ (B.T @ P)

# ─── nonlinear dynamics ─────────────────────────────────────────────────────────
def f(t, y):
    x, θ, x_dot, θ_dot = y
    s, c = np.sin(θ), np.cos(θ)
    u = (-K @ y)[0] if use_lqr else 0
    D  = M + m*s*s
    x_dd = (u + m*s*(l*θ_dot**2 + g*c) - b_cart*x_dot) / D
    θ_dd = ( -u*c - m*l*θ_dot**2*c*s - (M+m)*g*s + b_cart*x_dot*c - b_pend*θ_dot*l ) / (l*D)
    return [x_dot, θ_dot, x_dd, θ_dd]

# ─── simulate with fixed-step RK45 ──────────────────────────────────────────────
t0, tf, N = 0, 10, 800
t = np.linspace(t0, tf, N)
y0 = [0, np.deg2rad(90), 0, 0]
sol = solve_ivp(f, (t0, tf), y0, t_eval=t, method='RK45', max_step=(tf-t0)/N)

# ─── animate ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.set(xlim=(-1,1), ylim=(-0.6,0.6))
ax.set_aspect('equal','box')
ax.set_title("LQR ON" if use_lqr else "OPEN-LOOP")
ax.axhline(-0.1, color='k')

cart = plt.Rectangle((0,-0.1),0.3,0.2,fc='0.3')
pend, = ax.plot([], [], 'o-', lw=2)
ax.add_patch(cart)

def init():
    cart.set_xy((-0.15,-0.1)); pend.set_data([],[])
    return cart, pend

def update(i):
    x, θ = sol.y[0,i], sol.y[1,i]
    cart.set_xy((x-0.15,-0.1))
    pend.set_data([x, x+ l*np.sin(θ)], [0, -l*np.cos(θ)])
    return cart, pend

Anim = FuncAnimation(fig, update, frames=N, init_func=init,
                     interval=20, blit=True)
plt.show()