import matplotlib.pyplot as plt
import numpy as np
from model import GameModel
import network_creation

# Parameters
size_network = 40
connectivity_prob = 0.15
random_seed = 123
model_steps = 5

Vl = 10
Vh_values = np.linspace(10, 20, 100)
p = 15

num_simulations = 10

# Function to calculate tipping threshold
def calculate_tipping_threshold(Vl, Vh, p):
    # (Vh - Vl - p)/ (2*p)
    return (-Vh + Vl + p)/ (2*p)

print(calculate_tipping_threshold(8,11,4))

# so if penalty is 0, everyone would transition, so you get a very high number
# if penalty is 100, nobody wants to transition, so you get a negative number on the percentage of greens
# Thus, I need to find a number, if 4 neighbours would have transitioned, i would do the same

"""
# Calculate tipping thresholds
tipping_thresholds = [calculate_tipping_threshold(Vl, Vh, p) for Vh in Vh_values]
average_adoption_percentages = []

for i, Vh in enumerate(Vh_values):
    total_adoption_percentage = 0
    for _ in range(num_simulations):
        G = network_creation.create_connected_network(size_network, connectivity_prob, random_seed, homophily=False,
                                                      homophily_strength=0.25)

        model = GameModel(num_agents=size_network, network=G, Vl=Vl, Vh=Vh, p=p, initial_strategy_prob=0.1,
                          homophily=False)

        for step in range(model_steps):
            model.step()

        total_adoption_percentage += model.pct_norm_abandonmnet[-1]

        # Calculate average adoption percentage
    average_adoption_percentage = total_adoption_percentage / num_simulations
    average_adoption_percentages.append(average_adoption_percentage)


plt.plot(tipping_thresholds, average_adoption_percentages, marker='o')
plt.xlabel('Tipping Threshold')
plt.ylabel('Average Adoption Percentage')
plt.title('Average Adoption Percentage vs Tipping Threshold')
plt.show()

"""