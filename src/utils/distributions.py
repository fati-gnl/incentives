import numpy as np
import matplotlib.pyplot as plt
from src.model import GameModel
import random

# Parameters
Vh = 15
Vl = 9
num_samples = 1000

# Generate samples
#values = np.no_gamma.normal(loc=Vh, scale=0.5, size=num_samples)
#values = GameModel.generate_truncated_normal(mean=15, lower_bound=13, upper_bound=17, size=1000)
# values = np.no_gamma.normal(loc=Vh, scale=Vl)
values = np.random.normal(loc=Vh, scale=Vh - Vl, size=num_samples)

# Plot histogram
plt.hist(values, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()