import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Load Data ===
filename = "Double-Inverted-Pendulum-Cart-Control/RealLifeParameterEstimation/SpeedData.txt"
df = pd.read_csv(filename, sep='\t', header=None)

pwm_values = df.iloc[0].astype(float).values
velocity_data = df.iloc[1:].astype(float).values / 1000.0  # mm/s → m/s
time_step = 0.005  # seconds
max_voltage = 12.0  # volts
mass_kg = 0.1  # optional, for force estimation

# === Fit and Plot ===
plt.figure(figsize=(12, 8))
fit_results = []

print("\n=== Continuous-Time Model Coefficients and Fit Quality ===")
print(f"{'PWM':>5} {'V':>6} {'alpha':>10} {'beta':>10} {'gamma':>10} {'R²':>8}")

for idx, pwm in enumerate(pwm_values):
    trace = velocity_data[:, idx]
    trace = trace[~np.isnan(trace)]

    if len(trace) < 2:
        continue

    # Finite difference to approximate dv/dt
    v_k = trace[:-1]
    v_k1 = trace[1:]
    dv_dt = (v_k1 - v_k) / time_step

    # Convert PWM to Voltage
    voltage = pwm * max_voltage / 255.0
    V_k = np.full_like(v_k, voltage)

    # Regression: dv/dt = -alpha * v + beta * V + gamma
    X = np.column_stack([v_k, V_k, np.ones_like(v_k)])
    y = dv_dt

    reg = LinearRegression(fit_intercept=False).fit(X, y)
    neg_alpha, beta, gamma = reg.coef_
    alpha = -neg_alpha
    
    # Simulate forward first
    v_sim = [trace[0]]
    for _ in range(1, len(trace)):
        dv = -alpha * v_sim[-1] + beta * voltage + gamma
        v_sim.append(v_sim[-1] + dv * time_step)

    # Compute R² between real and simulated velocity
    r2 = r2_score(trace, v_sim)

    # Save results
    fit_results.append({
        'PWM': pwm,
        'Voltage': voltage,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'R2': r2
    })

    print(f"{int(pwm):5} {voltage:6.2f} {alpha:10.5f} {beta:10.5f} {gamma:10.5f} {r2:8.4f}")

    # Simulate forward with Euler
    v_sim = [trace[0]]
    for _ in range(1, len(trace)):
        dv = -alpha * v_sim[-1] + beta * voltage + gamma
        v_sim.append(v_sim[-1] + dv * time_step)

    t = np.arange(len(trace)) * time_step
    plt.plot(t, trace, 'k:', label=f"{int(pwm)} Meas.")
    plt.plot(t, v_sim, label=f"{int(pwm)} Fit")

# === Final Plot Formatting ===
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Continuous-Time Motor Model Fit")
plt.legend(fontsize=8, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def simulate_motor(alpha, beta, gamma, voltage_profile, time_step=0.005, initial_velocity=0.0):
    """
    Simulate motor dynamics using: dv/dt = -alpha*v + beta*V + gamma

    Args:
        alpha, beta, gamma: motor model parameters
        voltage_profile: array of input voltages (V) over time
        time_step: simulation time step (seconds)
        initial_velocity: starting velocity (m/s)

    Returns:
        time, velocity: arrays of time and velocity
    """
    n_steps = len(voltage_profile)
    v = np.zeros(n_steps)
    v[0] = initial_velocity

    for k in range(1, n_steps):
        dv = -alpha * v[k - 1] + beta * voltage_profile[k] + gamma
        v[k] = v[k - 1] + dv * time_step

    time = np.arange(n_steps) * time_step
    return time, v

# === Example usage ===

# Use example parameters from your fit
alpha = 31.0     # 1/s
beta = -0.3      # (m/s²)/V
gamma = -0.03    # m/s²

# Create a voltage step input (e.g., 0V for 0.2s, then 10V for 0.8s)
duration = 1.0
dt = 0.005
steps = int(duration / dt)
voltage_profile = np.zeros(steps)
voltage_profile[int(0.2 / dt):] = 8.0  # Step to 10V after 0.2s

# Simulate
t, v = simulate_motor(alpha, beta, gamma, voltage_profile, time_step=dt)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(t, v, label="Simulated Velocity (m/s)")
plt.plot(t, voltage_profile / max(voltage_profile) * max(v), '--', label="Voltage (scaled)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Motor Response to Voltage Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()