"""
distributions.py

With the use of this file, you can plot how different distributions of the Vh values would look like.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from src.model import GameModel
import random

# Parameters
Vh = 15
Vl = 9
num_samples = 1000

# Generate samples
def generate_truncated_normal(mean: list, lower_bound: float, upper_bound: float, size: int, sd: list) -> np.ndarray:
    """
    Generate values from a truncated normal distribution.
    :param [float] mean: Mean value for the distribution.
    :param float lower_bound: Lower bound for the truncated distribution.
    :param float upper_bound: Upper bound for the truncated distribution.
    :param int size: Number of values to generate.
    :param [float] sd: Standard deviation
    :return: Generated values from the truncated normal distribution.
    """

    a1, b1 = (lower_bound - mean[0]) / sd[0], (upper_bound - mean[0]) / sd[0]
    values = truncnorm.rvs(a1, b1, loc=mean[0], scale=sd[0], size=size)

    if len(mean) == 2:
        a2, b2 = (lower_bound - mean[1]) / sd[1], (upper_bound - mean[1]) / sd[1]
        values2 = truncnorm.rvs(a2, b2, loc=mean[1], scale=sd[1], size=size)

        # Combine values from both modes
        values = np.concatenate((values, values2))

    return values

values = np.random.uniform(low=9, high=13, size=100000)

# Plot histogram
plt.hist(values, bins=30, density=True, alpha=0.7, color='blue')
plt.tick_params(direction="in")
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
