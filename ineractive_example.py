import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from DIPC.config import DIPCParams
from DIPC.model import DIPC
from DIPC.trajectory_opt import solve_swingup_ocp
from DIPC.control import get_lqr, make_u_lqr

# note to the user - the plotting part of this code could be refactored

params = DIPCParams() # assign the parameters 

# define the inital state and the final state
x0 = np.array([-0.5, -np.pi/2, 0, 0, 0, 0])
xf = np.array([ 0.0,  np.pi/2, 0, 0, 0, 0])

t_swing = 3.0 # define the time you want for the swing up to take 
N = 300 # define the number of samples / 3 seconds

# call solve_swingup_ocp from the trajecotry optimization file and define the time
# grid and the force series
tgrid_swing, u_swing = solve_swingup_ocp(
    params, x0, xf, t_swing, N=N,
    R = np.array([[0.01]])
)

# Use DIPC to build the dynamics for the model 
# then linearize this model about the final (inverted) state 
model = DIPC(params)
f = model.f
A, B = model.linearize(xf, np.zeros((1,)))   # A, B continuous‐time

# use helper functions get_lqr and make_u_lqr in the 
# control file to make the controller
Q_lqr = np.diag([10,10,10,1,1,1])
R_lqr = np.array([[1.0]])
K_lqr = get_lqr(A, B, Q=Q_lqr, R=R_lqr)
u_lqr_fun = make_u_lqr(K_lqr, x_eq=xf, u_eq=np.array([0.0]))

# define a maximum force output on the cart
u_max_lqr = 100.0

# interactive state stuff
x = x0.copy()
t_sim = 0.0
dt = t_swing / (N - 1)
manual_force = 0.0
manual_Fmag = 20.0

# turning all flags off
flag_swing = False
flag_lqr   = False
swing_t0   = 0.0

# making limits for the cart so you cant lose it
x_min, x_max = -2.0, 2.0

def u_open_loop(t):
    idx = int((t - swing_t0) / dt)
    if 0 <= idx < len(u_swing):
        return float(u_swing[idx,0])
    return 0.0

# event handler stuff for interaction
def on_key_press(event):
    global flag_swing, swing_t0, flag_lqr, manual_force, x, t_sim
    
    # define the type of event
    k = event.key.lower()

    # large if statement that means you can toggle things on and off like
    # lqr, swing up, reset it, etc
    if k == 'r':
        flag_swing = not flag_swing
        swing_t0 = t_sim
        print(f"Swing-up {'ON' if flag_swing else 'OFF'}")

    elif k == 't':
        flag_lqr = not flag_lqr
        print(f"LQR {'ON' if flag_lqr else 'OFF'}")

    elif k == 'w':
        x = x0.copy()
        t_sim = 0.0
        swing_t0 = 0.0
        manual_force = 0.0
        flag_swing = False
        flag_lqr = False
        print("Simulation reset")

    elif k in ('left','a'):
        manual_force = -manual_Fmag

    elif k in ('right','d'):
        manual_force = manual_Fmag

# stop the forcing on key lift
def on_key_release(event):
    global manual_force
    if event.key.lower() in ('left','right','a','d'):
        manual_force = 0.0

# r4k integrator
def rk4_step(x, u):
    k1 = np.array(f(x,    [u])).flatten()
    k2 = np.array(f(x+dt/2*k1,[u])).flatten()
    k3 = np.array(f(x+dt/2*k2,[u])).flatten()
    k4 = np.array(f(x+dt*k3,  [u])).flatten()
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# settignu up the figure, using the xmin and max to get the total width
fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(x_min-0.5, x_max+0.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal') # no distortion

# same stuff as the plot function pretty much
cart_line, = ax.plot([], [], 'k', lw=6)
link1_line, = ax.plot([], [], 'r-', lw=2)
link2_line, = ax.plot([], [], 'b-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# these are the connectors from the events to the actual plot 
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# recognize this, another animate update functuon
def update(_):
    global x, t_sim

    # build control from flags
    u_ctrl = 0.0

    # checking flags to update the plot
    if flag_swing and swing_t0 <= t_sim < swing_t0 + t_swing:
        u_ctrl += u_open_loop(t_sim)

    if flag_lqr:
        # get array of shape (1,) then extact scalar
        raw_lqr = u_lqr_fun(x)
        raw_lqr = float(raw_lqr)
        u_ctrl  += np.clip(raw_lqr,
                           -u_max_lqr,
                           u_max_lqr)

    u_tot = u_ctrl + manual_force # float

    # integrate,  advance
    x = rk4_step(x, u_tot)
    t_sim += dt

    # enforce those cart lims cart limits
    x[0] = np.clip(x[0], x_min, x_max)

    # redraw things very similar to plot methods
    
    cx, θ1, θ2 = x[0], x[1], x[2]
    base = np.array([cx, 0.0])
    L1, L2 = params.L1, params.L2
    p1 = base + np.array([L1*np.cos(θ1), L1*np.sin(θ1)])
    p2 = p1 + np.array([L2*np.cos(θ1+θ2), L2*np.sin(θ1+θ2)])

    cart_line.set_data([cx-0.2, cx+0.2], [0,0])
    link1_line.set_data([base[0], p1[0]], [base[1], p1[1]])
    link2_line.set_data([p1[0],  p2[0]], [p1[1],  p2[1]])

    mode = ('S' if flag_swing else '-') + ('L' if flag_lqr else '-')
    time_text.set_text(f"t={t_sim:.2f}s  mode={mode}")
    force_text.set_text(f"u={u_tot:.1f} N")

    return cart_line, link1_line, link2_line, time_text, force_text

ani = FuncAnimation(
    fig, update,
    interval=dt*1000,
    blit=True,
    cache_frame_data=False
)

plt.title("←/A →/D = manual;  R = Swing Up;  T = LQR;  W = reset")
plt.show()