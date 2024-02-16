import numpy as np
import matplotlib.pyplot as plt

# Constants
Cm = 1.0  # Membrane capacitance (microfarads/cm^2)
g_Na = 120.0  # Sodium conductance (mS/cm^2)
g_K = 36.0  # Potassium conductance (mS/cm^2)
g_L = 0.3  # Leak conductance (mS/cm^2)
E_Na = 55.0  # Sodium reversal potential (mV)
E_K = -72.0  # Potassium reversal potential (mV)
E_L = -50.0  # Leak reversal potential (mV)
dt = 0.01  # Time step (ms)
t_max = 50.0  # Maximum simulation time (ms)

# Initial conditions
V_rest = -65.0  # Resting membrane potential (mV)
V = V_rest  # Initial membrane potential for one neuron

# Arrays to store results
time = np.arange(0, t_max, dt)
V_values_Na = np.zeros(len(time))  # Sodium spikes
V_values_K = np.zeros(len(time))  # Potassium spikes

# Simulation loop
for i, t in enumerate(time):
    # Stimulate the neuron with a current step for the first 5 ms
    I_inj = 10.0 if t < 5.0 else 0.0  # Injected current (nA/cm^2)

    # Hodgkin-Huxley model equations
    m = 1 / (1 + np.exp(-(V + 40.0) / 10.0))
    h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    n = 1 / (1 + np.exp(-(V + 55.0) / 10.0))

    # Ionic currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # Membrane potential dynamics
    dV_dt = (I_inj - I_Na - I_K - I_L) / Cm
    V += dV_dt * dt

    # Store the membrane potential for plotting
    V_values_Na[i] = I_Na
    V_values_K[i] = I_K

# Plotting
plt.figure(figsize=(12, 6))

# Plot sodium and potassium spikes
plt.plot(time, V_values_Na, label='Sodium')
plt.plot(time, V_values_K, label='Potassium', linestyle='--')

plt.xlabel('Time (ms)')
plt.ylabel('Ionic Current (nA/cm^2)')
plt.title('Sodium and Potassium Spikes (Single Neuron)')
plt.legend()
plt.grid(True)
plt.show()
