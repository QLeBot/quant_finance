import numpy as np
import random

# Define the states
states = ["Sunny", "Cloudy", "Rainy"]

# Define the transition matrix
# Rows: current state, Columns: next state
transition_matrix = [
    [0.6, 0.3, 0.1],  # Sunny -> [Sunny, Cloudy, Rainy]
    [0.2, 0.5, 0.3],  # Cloudy -> ...
    [0.1, 0.4, 0.5]   # Rainy -> ...
]

# Initial state
current_state = "Sunny"

# Map state names to indices
state_index = {state: idx for idx, state in enumerate(states)}

# Simulate the Markov chain for 15 days
np.random.seed(42)
n_days = 15
weather_sequence = [current_state]

for _ in range(n_days - 1):
    current_idx = state_index[current_state]
    next_state = np.random.choice(
        states,
        p=transition_matrix[current_idx]  # probabilities for the current state
    )
    weather_sequence.append(next_state)
    current_state = next_state

# Print the weather sequence
print("Weather over 15 days:")
print(" -> ".join(weather_sequence))
