import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Wear Model based on Archard’s Principles ---
# This differential equation represents the wear rate of a robot joint.
# dW/dt = 10^-6 * (1 + sin^2(t)) * (1 + 0.1W)
def wear_rate(t, W):
    return 1e-6 * (1 + np.sin(t)**2) * (1 + 0.1 * W)

# --- 2. Corrected Numerical Methods Implementation ---
def solve_ode(method, t_end, step_size):
    t_values = np.arange(0, t_end + step_size, step_size)
    # The fix: Initialize the list with the starting wear value (0 microns)
    w_values = [0.0] 
    
    w = 0.0
    h = step_size
    
    for i in range(len(t_values) - 1):
        t = t_values[i]
        
        if method == 'Euler':
            # Euler's Method: y_new = y + h * f(t, y)
            w = w + h * wear_rate(t, w)
            
        elif method == 'RK4':
            # 4th Order Runge-Kutta Method [3]
            k1 = h * wear_rate(t, w)
            k2 = h * wear_rate(t + 0.5*h, w + 0.5*k1)
            k3 = h * wear_rate(t + 0.5*h, w + 0.5*k2)
            k4 = h * wear_rate(t + h, w + k3)
            # Weighted average update
            w = w + (k1 + 2*k2 + 2*k3 + k4) / 6
            
        w_values.append(w)
        
    return t_values, w_values

# --- 3. Execute Comparison for BMW Maintenance Scheduling ---
# We simulate 2000 hours of operation (approx. 3 months of 3-shift production)
t_end = 2000 
step = 100   # Larger step size highlights the approximation error 

t_euler, w_euler = solve_ode('Euler', t_end, step)
t_rk4, w_rk4 = solve_ode('RK4', t_end, step)

# --- 4. Plotting Results for Chapter 4 ---
plt.figure(figsize=(12, 7))
plt.plot(t_euler, w_euler, 'r--o', label='Euler Method (Approximation)', markersize=4)
plt.plot(t_rk4, w_rk4, 'b-s', label='Runge-Kutta 4 (High Precision)', markersize=4)

# Adding BMW-specific failure threshold for context
failure_limit = 0.0030
plt.axhline(y=failure_limit, color='k', linestyle=':', label='Maintenance Threshold')

plt.title('Predictive Maintenance: Robot Joint Wear Forecast (BMW iFactory Context)')
plt.xlabel('Operation Time (Hours)')
plt.ylabel('Accumulated Wear (Microns)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Show final error quantification
print(f"--- Numerical Results at {t_end} Hours ---")
print(f"Predicted Wear (Euler): {w_euler[-1]:.6f} microns")
print(f"Predicted Wear (RK4):   {w_rk4[-1]:.6f} microns")
print(f"Absolute Divergence:    {abs(w_rk4[-1] - w_euler[-1]):.6f} microns")
print(f"Relative Approximation Error: {(abs(w_rk4[-1] - w_euler[-1])/w_rk4[-1])*100:.2f}%")

plt.show()
           
