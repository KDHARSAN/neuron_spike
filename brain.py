import brian2 as b2
import matplotlib.pyplot as plt

b2.prefs.codegen.target = "numpy"

tau = 20 * b2.ms  # Membrane time constant
v_rest = -70 * b2.mV  # Resting potential
v_threshold = -50 * b2.mV  # Threshold for spiking
v_reset = -65 * b2.mV  # Reset potential (closer to v_rest for bursting)
Ca_factor = 20 * b2.mV / b2.amp  # Factor to convert Ca from amps to volts


Ca = 0 * b2.amp  # Calcium-like variable (initially 0)
Ca_tau = 100 * b2.ms  # Calcium time constant

eqs = '''
dv/dt = (v_rest - v + Ca_factor * Ca) / tau + 10 * mV / ms : volt
dCa/dt = -Ca / Ca_tau : amp  # Use amp for current units
'''

neuron_group = b2.NeuronGroup(100, eqs, threshold='v > v_threshold', reset='v = v_reset; Ca += 0.5 * amp', method='euler')
state_monitor = b2.StateMonitor(neuron_group, ['v', 'Ca'], record=True)
spike_monitor = b2.SpikeMonitor(neuron_group)

b2.run(1000 * b2.ms)

print("State monitor data:", state_monitor.v, state_monitor.Ca)
print("Spike monitor times:", spike_monitor.t)


plt.figure(figsize=(12, 6))


"""plt.subplot(2, 1, 1)"""
plt.plot(state_monitor.t / b2.ms, state_monitor.v[0] / b2.mV, label='v')
plt.plot(state_monitor.t / b2.ms, state_monitor.Ca[0] / b2.amp, label='Ca')
plt.title('Membrane Potential and Calcium Dynamics')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV) / Calcium (amp)')
plt.legend()


"""plt.subplot(2, 1, 2)
plt.plot(spike_monitor.t / b2.ms, spike_monitor.i, '|k')
plt.title('Neuron Spike Raster')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
"""
plt.tight_layout()
plt.show()
