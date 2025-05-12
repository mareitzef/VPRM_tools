import numpy as np
import matplotlib.pyplot as plt
import math

# Provided Tscale function
def calculate_tscale(t, t_min, t_max, t_opt):
    if not math.isfinite(t):
        return 0

    if t_min <= t <= t_max:
        denominator = (t - t_min) * (t - t_max) - (t - t_opt) ** 2
        if denominator != 0:
            value = ((t - t_min) * (t - t_max)) / denominator
        else:
            value = 0
    else:
        value = 0

    return value

# Vectorized version for plotting
def vectorized_tscale(T, t_min, t_max, t_opt):
    return np.array([calculate_tscale(t, t_min, t_max, t_opt) for t in T])

# Constants
T_min = 0
T_max = 45
T_opt = 25

# Temperature range
T_range = np.linspace(-5, 50, 1000)
Tscale_values = vectorized_tscale(T_range, T_min, T_max, T_opt)

# Numerical derivative using central difference
def numerical_derivative(y, x):
    dy = np.gradient(y, x)
    return dy

# Constants
PAR0 = 1000  # Can be changed
lambda_ = 0.07  # Can be changed
PAR = 500  # Can be changed

# Recalculate Tscale and GPP for this setting
Tscale_simplified = vectorized_tscale(T_range, T_min, T_max, T_opt)
GPP_simplified = (lambda_ * Tscale_simplified / (1 + PAR / PAR0)) * PAR

# Numerical derivative of GPP with respect to temperature
dGPP_dT = numerical_derivative(GPP_simplified, T_range)

# Plot GPP and its derivative
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(T_range, GPP_simplified, label="GPP", color="green")
plt.axvline(T_opt, linestyle="--", color="red", label="Topt = 25°C")
plt.title("Simplified GPP vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("GPP")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(T_range, dGPP_dT, label="dGPP/dT", color="blue")
plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
plt.axvline(T_opt, linestyle="--", color="red", label="Topt = 25°C")
plt.title("Derivative of GPP vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("dGPP/dT")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
