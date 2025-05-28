"""
From youtube : https://www.youtube.com/watch?v=0x-Pc-Z3wu4

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import qfin as qf

def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2)) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return call_price

print(black_scholes_call(100, 100, 0.3, 0.05, 1))

def black_scholes_put(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2)) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    put_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

print(black_scholes_put(100, 100, 0.3, 0.05, 1))

path = qf.simulations.GeometricBrownianMotion(100, 0.05, 0.3, 1/252, 1)

plt.title("Terminal Value of an Option Contract")
plt.hlines(100, 0, 252, label='Strike', color='orange')
plt.plot(path.simulated_path, label='Price Path', color='blue')
if max(path.simulated_path[-1] - 100, 0) == 0:
    plt.vlines(252, path.simulated_path[-1], 100, color='red', label='P/L')
else:
    plt.vlines(252, 100, path.simulated_path[-1], color='green', label='P/L')
plt.style.use('dark_background')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print("Premium at t=0:", black_scholes_call(100, 100, 0.3, 0.05, 1))
print("P/L:", max(path.simulated_path[-1] - 100, 0) - black_scholes_call(100, 100, 0.3, 0.05, 1))

