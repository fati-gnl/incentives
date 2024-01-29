"""
Experiment1.py

Adoption of New Technology Over Time for Different Count Numbers

This experiment investigates the adoption of new technology over time for varying count numbers.
Different parameters such as network structure, gamma distribution, and initiator strategy can be adjusted.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.network_creation import create_connected_network
from src.model import GameModel

# Parameters
size_network = 1000
connectivity_prob = 0.5
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
p = 8

# Initialize list to store results
results = []

# Loop through different initiators values
for initiator in np.linspace(175, 350, 10, dtype=int):

    # Generate a connected no_gamma network
    G, node_degrees = create_connected_network(
        size_network, connectivity_prob, random_seed, Vh=Vh,homophily=False, homophily_strength=0.01,
        initiators=initiator,node_degree=0,gamma=True,initialisation="Lowest_degree",
        incentive_count = 0, incentive_amount = 0, incentive_strategy="Highest_degree")

    # Create the model
    model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, p=p)

    # Run the model for a certain number of steps
    for step in range(model_steps):
        model.step()

    # Save the percentage of agents adopting new technology for each step
    results.append((initiator, model.pct_norm_abandonmnet))

# Plotting
for initiator, pct_values in results:
    plt.plot(range(1, model_steps + 1), pct_values, marker='o', label=f"Count: {initiator}")

plt.xlabel("Step Number")
plt.ylabel("Percentage of Adoption")
plt.title("Adoption of New Technology Over Time for Different Initiator Numbers")
plt.legend()
plt.show()
