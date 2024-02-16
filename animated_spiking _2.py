import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
C_m = 1.0  # Membrane capacitance, in uF/cm^2
g_Na = 120.0  # Sodium (Na) maximum conductances, in mS/cm^2
g_K = 36.0  # Potassium (K) maximum conductances, in mS/cm^2
g_L = 0.3  # Leak maximum conductances, in mS/cm^2
E_Na = 50.0  # Sodium (Na) Nernst reversal potentials, in mV
E_K = -77.0  # Potassium (K) Nernst reversal potentials, in mV
E_L = -54.387  # Leak Nernst reversal potentials, in mV

# Define a function to generate Poisson spikes
def poisson_spikes(t, N=100, rate=1.0): 
    spks = []
    dt = t[1] - t[0]
    for n in range(N):                                                       #generates spikes using poisson process
        spkt = t[np.random.rand(len(t)) < rate*dt/1000.]
        idx = [n] * len(spkt)
        spkn = np.concatenate([[idx], [spkt]], axis=0).T
        if len(spkn) > 0:
            spks.append(spkn)
    spks = np.concatenate(spks, axis=0)
    return spks

# Set up the simulation parameters
N = 100                                #total number if neuron 
N_ex = 80  # (0..79)                   #This parameter represents the number of excitatory neurons in the network
N_in = 20  # (80..99)                  #This parameter represents the number of inhibitory neurons in the network
G_ex = 1.0                             #This parameter represents the excitatory synaptic conductance, which is a measure of how easily ions flow through the synapses between neurons. 
K = 4                                  #This parameter represents a scaling factor.
dt = 0.01                              #This parameter represents the time step used in the simulation
ic = [-65, 0.05, 0.6, 0.32]            #Initial condition for the simulation 
t = np.arange(0.0, 300.0, dt)          #time array
spks = poisson_spikes(t, N, rate=10.0)  

# Define the gating functions
def alpha_m(V):                                       
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):                                         
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):                                                        
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80.0)

# Define the current functions
def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)

def I_K(V, n):
    return g_K * n**4 * (V - E_K)

def I_L(V):
    return g_L * (V - E_L)

def I_app(t):
    return 3.0

def I_syn(spks, t):
    exspk = spks[spks[:, 0] < N_ex]
    delta_k = exspk[:, 1] == t
    if sum(delta_k) > 0:
        h_k = np.random.rand(len(delta_k)) < 0.5
    else:
        h_k = 0
    inspk = spks[spks[:, 0] >= N_ex]
    delta_m = inspk[:, 1] == t
    if sum(delta_m) > 0:
        h_m = np.random.rand(len(delta_m)) < 0.5
    else:
        h_m = 0
    isyn = C_m * G_ex * (np.sum(h_k * delta_k) - K * np.sum(h_m * delta_m))
    return isyn

# Define the differential equations
def dALLdt(X, t):
    V, m, h, n = X
    dVdt = (I_app(t) + I_syn(spks, t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

# Solve the differential equations
from scipy.integrate import odeint
X = odeint(dALLdt, ic, t)

# Set up the animation
fig, ax = plt.subplots(figsize=(8,5))

def update(frame):
        t_ini = frame * 100.0
        t_end = (frame + 1) * 101.0
        t_interval = np.logical_and(t > t_ini, t < t_end)
        line.set_xdata(t[t_interval])
        line.set_ydata(X[t_interval, 0])
        ax.set_title('Neuron Membrane Potential over Time')
        ax.set_xlabel('Time (ms)')
        ax.set_facecolor('black') 
        ax.set_ylabel('Membrane Potential (mV)')
        ax.grid(True, linestyle='--', alpha=0.8)
        return line,

line,= ax.plot(t, X[:, 0], label='Membrane Potential (mV)',color='white', linewidth=1)
ani = animation.FuncAnimation(fig=fig, func=update, frames=3, interval=100)
plt.show()
plt.close()