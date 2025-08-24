
# section for pivot friction coeff est

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# === 1. Load and preprocess data ===
filename = "Double-Inverted-Pendulum-Cart-Control/RealLifeParameterEstimation/Friction/PendulumFriction/90DegFreeOsc.csv"
data = pd.read_csv(filename, header=None)
theta = data[0].values

dt = 0.01  # samping @ 100hz 
t = np.arange(len(theta)) * dt 

# === 2. Find peaks ===
peaks, _ = find_peaks(theta, distance=50)
peak_times = t[peaks]
peak_values = theta[peaks]

# === Fit exponential decay to peaks ===
def exp_decay(t, A, lamb):
    return A * np.exp(-lamb * t)

# Take absolute value in case peaks are negative
popt, _ = curve_fit(exp_decay, peak_times, np.abs(peak_values))
A_fit, lamb_fit = popt

# === Estimate damping ratio and b_pend ===
# Parameters
m = 0.035    # kg
l = 0.075    # m
g = 9.81     # m/s²
I = m * l**2 
omega_n = np.sqrt(m * g * l / I)

zeta = lamb_fit / omega_n
b_pend = 2 * zeta * np.sqrt(I * m * g * l)

# === Print results ===
print(f"Fitted exponential decay rate λ: {lamb_fit:.4f} 1/s")
print(f"Natural frequency ω_n: {omega_n:.4f} rad/s")
print(f"Damping ratio ζ: {zeta:.4f}")
print(f"Estimated b_pend: {b_pend:.6f} N·m·s")

# === Plot ===
plt.plot(t, theta, label="Angle")
plt.plot(peak_times, peak_values, 'ro', label="Peaks")
plt.plot(peak_times, exp_decay(peak_times, *popt), 'k--', label="Exponential Fit")
plt.xlabel("Time (s)")
plt.ylabel("Angle")
plt.title("Pendulum Oscillation with Exponential Envelope Fit")
plt.legend()
plt.grid(True)
plt.show()