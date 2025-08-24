import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

filename = "Double-Inverted-Pendulum-Cart-Control/RealLifeParameterEstimation/SpeedData.txt"
df = pd.read_csv(filename, sep='\t', header = None)
# print(df)

# get the data into a numpy array 
pwm_values = df.iloc[0].astype(float).values 
velocity_data = df.iloc[1:].astype(float).values
velocity_data /= 1000.0 # the velocity data is in mm/s, I want to use m/s
time_step = 0.005 # this is the data collection rate used

plt.figure(figsize=(12,8))

fit_results = []

for idx, pwm in enumerate(pwm_values):
    trace = velocity_data[:, idx]
    trace = trace[~np.isnan(trace)]

    v_k = trace[:-1]
    v_k1 = trace[1:]

    # Regression: v_k1 = a * v_k + b * u + c
    X = np.column_stack((v_k, np.ones_like(v_k)))
    u_vec = np.full_like(v_k, pwm)
    X = np.column_stack((X, u_vec))
    y = v_k1

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)

    a, c, b = reg.coef_

    # Simulate
    model_pred = [trace[0]]
    for i in range(1, len(trace)):
        model_pred.append(a * model_pred[-1] + b * pwm + c)

    t = np.arange(len(trace)) * time_step
    plt.plot(t, trace, linestyle='dotted', label=f"{int(pwm)} Meas.")
    plt.plot(t, model_pred, linestyle='solid', label=f"{int(pwm)} Fit")

    r2 = r2_score(v_k1, reg.predict(X))
    fit_results.append({'PWM': pwm, 'a': a, 'b': b, 'c': c, 'R2': r2})

# === Show fit plot ===
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Motor Model with Friction Offset")
plt.legend(fontsize=8, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

for res in fit_results:
    print(f"PWM {int(res['PWM']):3} | a = {res['a']:.5f}, b = {res['b']:.5f}, c = {res['c']:.5f}, R² = {res['R2']:.4f}")

# === Plot b vs PWM ===
pwms = [res['PWM'] for res in fit_results]
bs = [res['b'] for res in fit_results]

plt.figure(figsize=(6, 4))
plt.plot(pwms, bs, 'o-')
plt.xlabel("PWM Value")
plt.ylabel("b Coefficient (Input Gain)")
plt.title("b vs PWM")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Force estimate with friction included ===
mass_kg = 0.1
mm_to_m = 1
dt = 0.005

F_newtons = [mass_kg * (b * pwm + c) * mm_to_m / dt for pwm, b, c in zip(pwms, bs, [res['c'] for res in fit_results])]

plt.figure(figsize=(6, 4))
plt.plot(pwms, F_newtons, 'o-')
plt.xlabel("PWM Value")
plt.ylabel("Estimated Force (N)")
plt.title("Force Output vs PWM (Friction Compensated)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n=== Estimated Forces with Friction ===")
for pwm, F in zip(pwms, F_newtons):
    print(f"PWM {int(pwm)} → Force ≈ {F:.2f} N")