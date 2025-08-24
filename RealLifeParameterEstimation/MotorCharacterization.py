import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Load manually-trimmed data ===
filename = "Double-Inverted-Pendulum-Cart-Control/RealLifeParameterEstimation/SpeedData.txt"  # Replace with your file path
df = pd.read_csv(filename, sep='\t', header=None)

pwm_values = df.iloc[0].astype(float).values
velocity_data = df.iloc[1:].astype(float).values  # shape: (timesteps, num_pwms)
time_step = 0.005  # seconds

# === Fit and simulate each trial independently ===
plt.figure(figsize=(12, 8))

fit_results = []

for idx, pwm in enumerate(pwm_values):
    trace = velocity_data[:, idx]
    trace = trace[~np.isnan(trace)]  # drop any trailing NaNs due to manual trim

    v_k = trace[:-1]
    v_k1 = trace[1:]

    # Regression: v_k1 = a * v_k + b * u  --> fit intercept = b*u
    X = v_k.reshape(-1, 1)
    y = v_k1
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    a = reg.coef_[0]
    b_prime = reg.intercept_
    b = b_prime / pwm if pwm != 0 else 0.0

    # Simulate using fitted model
    model_pred = [trace[0]]
    for i in range(1, len(trace)):
        model_pred.append(a * model_pred[-1] + b * pwm)

    t = np.arange(len(trace)) * time_step
    plt.plot(t, trace, linestyle='dotted', label=f"{int(pwm)} Meas.")
    plt.plot(t, model_pred, linestyle='solid', label=f"v[k+1]={a:.3f}*v[k]+{b:.3f}*{int(pwm)}")

    # Compute R^2
    r2 = r2_score(v_k1, reg.predict(X))

    # Store for summary
    fit_results.append({
        'PWM': pwm,
        'a': a,
        'b': b,
        'R2': r2
    })

    # Print to terminal
    print(f"[PWM {int(pwm)}] a = {a:.5f}, b = {b:.5f}, R² = {r2:.4f}")

# === Show final plot ===
plt.xlabel("Time (s)")
plt.ylabel("Velocity (mm/s)")
plt.title("Motor Model Fits Per Trial (Manual Trimmed Data)")
plt.legend(fontsize=8, loc='lower right')  # Legend position here
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n=== Summary Table ===")
for result in fit_results:
    print(f"PWM {int(result['PWM']):3} | a = {result['a']:.5f}, b = {result['b']:.5f}, R² = {result['R2']:.4f}")
    

pwms = [res['PWM'] for res in fit_results]
bs = [res['b'] for res in fit_results]

plt.figure(figsize=(6, 4))
plt.plot(pwms, bs, 'o-')
plt.xlabel("PWM Value")
plt.ylabel("Estimated b Coefficient")
plt.title("Motor Input Gain (b) vs PWM")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Compute and plot corrected motor force in Newtons ===

# Pull PWM and b values from fit results
pwm_vals = [res['PWM'] for res in fit_results]
b_vals   = [res['b'] for res in fit_results]

mass_kg = 0.1
dt = 0.005  # 5 ms time step
mm_to_m = 1 / 1000  # convert mm to m

F_newtons = [mass_kg * (b * pwm) * mm_to_m / dt for b, pwm in zip(b_vals, pwm_vals)]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(pwm_vals, F_newtons, 'o-', label='Force (N)')
plt.xlabel("PWM Value")
plt.ylabel("Estimated Force (Newtons)")
plt.title("Corrected Motor Force Output vs PWM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print representative values
print("\n=== Corrected Force in Newtons ===")
for pwm, F in zip(pwm_vals, F_newtons):
    print(f"PWM {int(pwm)} → Force ≈ {F:.2f} N")