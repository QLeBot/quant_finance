"""
Brownian motion is a continuous-time stochastic process that describes the random movement of particles in a fluid.
It is named after the botanist Robert Brown, who first observed the movement of pollen particles under a microscope in 1827.

Key properties of Brownian motion:
1. It is a Markov process, meaning that the future movement depends only on the current position, not on the path taken to reach it.


"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate Brownian motion
T = 1.0         # total time
N = 500         # number of steps
dt = T / N      # time step
t = np.linspace(0, T, N)
B = np.zeros(N)
B[1:] = np.cumsum(np.sqrt(dt) * np.random.randn(N - 1))  # sum of normal steps

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, B, label="Brownian Motion Path")
plt.xlabel("Time")
plt.ylabel("B(t)")
plt.title("Simulated Brownian Motion")
plt.grid(True)
plt.legend()
plt.show()
