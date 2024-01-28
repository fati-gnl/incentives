import networkx as nx
import network_creation
from model import GameModel
import matplotlib.pyplot as plt
import numpy as np

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh=11
Vl = 8
p = 8

# Generate a connected no_gamma network
G, node_degrees = network_creation.create_connected_network(size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01, count=10, node_degree=0, gamma=False, initialisation="Highest_node")

# Create the model
model = GameModel(num_agents=size_network, network=G, node_degrees= node_degrees, Vl=Vl, Vh=Vh, p=p)

# Run the model for a certain number of steps
for step in range(model_steps):
    model.step()

    # Get the network with colors
    #network_data = model.get_network_with_colors()

    # Plot the network
    #plt.figure()
    #plt.title(f'Step {step}')
    #nx.draw(network_data["graph"], network_data["pos"], node_color=network_data["colors"], labels=network_data["alpha_values"])
    #plt.show()

# Plotting
plt.plot(range(1, model_steps + 1), model.pct_norm_abandonmnet, marker='o')
plt.xlabel('Step Number')
plt.ylabel('Percentage Adopt New Technology')
plt.title('Percentage of Agents Adopting New Technology Over Time')
plt.show()

# Initialize list to store results
#results = []

# Loop through different initial degrees
#for initial_degree in np.unique(list(node_degrees.values())):
    #pct_abandonment = model.run_model_for_initial_degree(initial_degree)
    #results.append((initial_degree, pct_abandonment))

#x_values, y_values = zip(*results)

# Plotting
#plt.plot(x_values, y_values, marker='o')
#plt.xlabel("Initial Node Degree")
#plt.ylabel("Percentage of Adoption")
#plt.title("Adoption of New Technology vs. Initial Node Degree")
#plt.show()