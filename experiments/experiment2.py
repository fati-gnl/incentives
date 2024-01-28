"""
Experiment2.py

Cascade Size as a Function of Initiator Probability

This experiment investigates the cascade size as a function of the nodes whose strategy is "Adopt New Technology" at the
start of the simulation. Different parameters such as network structure, gamma distribution, and initiator strategy can be adjusted.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.network_creation import create_connected_network
from src.model import GameModel

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
p = 8

# Initialize list to store results
p_values = [5,6,8,10]
results = {p: [] for p in p_values}

# Loop through different initiator probabilities
initiator_probs = np.linspace(0.1, 0.45, 25)

for p in p_values:
    for initiator_prob in initiator_probs:
        G, node_degrees = create_connected_network(
            size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
            count=(int(initiator_prob * size_network)), node_degree=0, gamma=False, initialisation="higuest_node")

        model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, p=p)

        for step in range(model_steps):
            model.step()

        final_cascade_size = model.get_final_cascade_size_scaled() / size_network

        results[p].append((initiator_prob, final_cascade_size))

def calculate_tipping_threshold(Vl, Vh, p):
    return (-Vh + Vl + p)/ (2*p)

for p in p_values:
    tipping_threshold = calculate_tipping_threshold(Vl, Vh, p)
    plt.plot(initiator_probs, [cascade_size for initiator_prob, cascade_size in results[p]], marker='o', label=str(tipping_threshold))

plt.xlabel("Initiator Probability")
plt.ylabel("Cascade Size")
plt.title("Cascade Size as a Function of Initiator Probability")
plt.legend()
plt.show()