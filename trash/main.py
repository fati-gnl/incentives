import networkx as nx
from src.network_creation import create_connected_network
from src.model import GameModel
import matplotlib.pyplot as plt
import numpy as np

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 25
Vh=11
Vl = 8
p = 8

initiators = 0.20

incentive_strategy = "Random"
initialisation = "Random"
incentive_count_values = np.linspace(0.1, 0.80, 20)

incentive_count =  int(0.1 * size_network)
incentive_amount = 100000

# Generate a connected no_gamma network
G, node_degrees = create_connected_network(
            size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
            initiators=int(initiators*size_network), node_degree=0, gamma=True,
            initialisation=initialisation, incentive_count=incentive_count, incentive_amount=incentive_amount,
            incentive_strategy=incentive_strategy)

# Create the model
model = GameModel(num_agents=size_network, network=G, node_degrees= node_degrees, Vl=Vl, p=p)

# Run the model for a certain number of steps
for step in range(model_steps):
    model.step()

    # Get the network with colors
    network_data = model.get_network_with_colors()

    # Plot the network
    plt.figure()
    plt.title(f'Step {step}')
    nx.draw(network_data["graph"], network_data["pos"], node_color=network_data["colors"], labels=network_data["gamma_values"])
    plt.show()

# Plotting
plt.plot(range(1, model_steps + 1), model.pct_norm_abandonmnet, marker='o')
plt.xlabel('Step Number')
plt.ylabel('Percentage Adopt New Technology')
plt.title('Percentage of Agents Adopting New Technology Over Time')
plt.show()

# Initialize list to store figures
#figures = []

# Loop through different initial degrees
#for initial_degree in np.unique(list(node_degrees.values())):
    #pct_abandonment = model.run_model_for_initial_degree(initial_degree)
    #figures.append((initial_degree, pct_abandonment))

#x_values, y_values = zip(*figures)

# Plotting
#plt.plot(x_values, y_values, marker='o')
#plt.xlabel("Initial Node Degree")
#plt.ylabel("Percentage of Adoption")
#plt.title("Adoption of New Technology vs. Initial Node Degree")
#plt.show()