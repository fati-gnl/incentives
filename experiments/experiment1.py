import matplotlib.pyplot as plt
import numpy as np
import src.network_creation
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
results = []

# Loop through different count values
for count in np.linspace(319, 319, 1, dtype=int):

    # Generate a connected no_gamma network
    G, node_degrees = network_creation.create_connected_network(
        size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
        count=count, node_degree=0
    )
    G, node_degrees = network_creation.create_connected_network(
        size_network, connectivity_prob, random_seed, Vh=Vh,homophily=False, homophily_strength=0.01,
        count=count,node_degree=0,gamma=False,initialisation="higuest_node")

    # Create the model
    model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, Vh=Vh, p=p)

    # Run the model for a certain number of steps
    for step in range(model_steps):
        model.step()

    # Save the percentage of agents adopting new technology for each step
    results.append((count, model.pct_norm_abandonmnet))

# Plotting
for count, pct_values in results:
    plt.plot(range(1, model_steps + 1), pct_values, marker='o', label=f"Count: {count}")

plt.xlabel("Step Number")
plt.ylabel("Percentage of Adoption")
plt.title("Adoption of New Technology Over Time for Different Count Numbers")
plt.legend()
plt.show()
