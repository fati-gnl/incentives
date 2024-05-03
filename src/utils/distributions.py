"""
distributions.py

With the use of this file, you can plot how different distributions of the Vh values would look like.
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from src.model import GameModel

matplotlib.use('Agg')

# Parameters
Vh = 11
Vl = 9
num_samples = 1000

values = np.random.choice(
        GameModel.generate_distribution(mean=11, min_value=8,width_percentage=0.675,entitled_distribution = "BiModal"), size=1000)

#values = np.random.normal(Vh, 0.5, 30000)
#values = np.random.uniform(Vh-2, Vh+2, 30000)

# Plot histogram
plt.hist(values, bins=30, density=True, alpha=0.7, color='blue')
plt.tick_params(direction="in")
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.savefig("d1")
