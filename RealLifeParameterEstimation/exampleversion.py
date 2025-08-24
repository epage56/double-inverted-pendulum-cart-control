import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp

# === Load and parse the SpeedData.txt file ===
file_path = "Double-Inverted-Pendulum-Cart-Control/RealLifeParameterEstimation/SpeedData.txt"
df = pd.read_csv(file_path, sep='\t', header=None)

# First row is voltage
voltages = df.iloc[0].astype(float).values
# Remaining rows are velocity data
velocities = df.iloc[1:].astype(float).values

# Simulate the 'measures' format: [voltage, pos0, dt0, pos1, dt1, ...]
time_step_us = 5000  # 0.005 seconds = 5000 µs
measures = []

for i, voltage in enumerate(voltages):
    trace = velocities[:, i]
    trace = trace[~np.isnan(trace)]

    if len(trace) < 2:
        continue

    positions = [0.0]
    for v in trace[1:]:
        new_pos = positions[-1] + v * 0.005  # m/s to mm
        positions.append(new_pos)

    interleaved = []
    for pos in positions:
        interleaved.append(pos)
        interleaved.append(time_step_us)

    measures.append([voltage] + interleaved)

# === Analysis begins ===

colormap = plt.cm.prism
plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, len(measures))])
plt.ylabel("Cart speed, m/s")
plt.xlabel("Time, seconds")

u     = []
w0    = []
alpha = []

for m in reversed(range(len(measures))):
    voltage = measures[m][0]
    u.append(voltage)

    data = measures[m][1:]
    position = data[0::2]
    deltatime = data[1::2]

    time = []
    speed = []
    seconds = 0.0

    for i in range(1, len(deltatime)):
        dt = deltatime[i - 1] / 1e6
        seconds += dt
        time.append(seconds)
        v = (position[i] - position[i - 1]) / 1000.0 / dt
        speed.append(v)

    cnt = 0
    w0.append(0)
    for i in range(1, len(speed) // 10):
        w0[-1] += speed[-i]
        cnt += 1
    w0[-1] /= cnt

    alpha.append(-1)
    best_energy = float("inf")
    for a in np.arange(-100, 0, 0.1):
        energy = sum((w0[-1] * (1 - exp(a * t)) - v) ** 2 for t, v in zip(time, speed))
        if energy < best_energy:
            alpha[-1] = a
            best_energy = energy

    fit = [w0[-1] * (1 - exp(alpha[-1] * t)) for t in time]
    plt.plot(time, speed, label="%.1fV" % voltage)
    # plt.plot(time, fit)

# === Second Phase Fit ===
(a, b, c) = (0, 0, 0)
best_energy = float("inf")

for ta in np.arange(0.1, 1.0, 0.001):
    A = [[u[m], np.sign(u[m])] for m in range(len(u))]
    B = [w0[m] - ta * w0[m] for m in range(len(u))]

    A = np.array(A)
    B = np.array(B)
    AtA = A.T @ A
    AtB = A.T @ B
    tb, tc = np.linalg.solve(AtA, AtB)

    energy = 0
    for m in range(len(u)):
        wk = 0
        for t in np.arange(0, 0.5, 0.02):
            desired = w0[m] * (1 - exp(alpha[m] * t))
            energy += (desired - wk) ** 2
            wk = ta * wk + tb * u[m] + tc * np.sign(u[m])
    if energy < best_energy:
        best_energy = energy
        (a, b, c) = (ta, tb, tc)

# === Final Simulation Plot ===
for m in range(len(u)):
    A_vals = []
    T = []
    wk = 0
    for t in np.arange(0, 0.5, 0.02):
        A_vals.append(wk)
        T.append(t)
        wk = a * wk + b * u[m] + c * np.sign(u[m])
    plt.plot(T, A_vals)
    # plt.plot(T, [w0[m]*(1-exp(alpha[m]*t)) for t in T])  # Optional overlay

print("Best-fit parameters: a=%.3f, b=%.3f, c=%.3f" % (a, b, c))

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()