"""
Predicted_norm_abandonment.py
With the use of this file, you can calculate the predicted tipping threshold for a particular Vl, Vh and p value.
"""
import numpy as np

# Parameters
size_network = 40
connectivity_prob = 0.15
random_seed = 123
model_steps = 5
Vl = 3
Vh_values = np.linspace(4, 8, 100)
p = 2
num_simulations = 10

# Function to calculate tipping threshold
def calculate_tipping_threshold(Vl, Vh, p):
    return (-Vh + Vl + p)/ (2*p)

print(calculate_tipping_threshold(3,6,4))