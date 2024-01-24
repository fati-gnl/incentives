import networkx as nx
import network_creation
from model import GameModel
import matplotlib.pyplot as plt

# Parameters
size_network = 20
connectivity_prob = 0.15
random_seed = 123
model_steps = 10

# Generate a connected random network
G = network_creation.create_connected_network(size_network, connectivity_prob, random_seed)

# Create the model
model = GameModel(num_agents=size_network, network=G, Vl=9, Vh=15, p=5, initial_strategy_prob=0.1)

# Run the model for a certain number of steps
for step in range(model_steps):
    model.step()

    # Get the network with colors
    network_data = model.get_network_with_colors()

    # Plot the network
    plt.figure()
    plt.title(f'Step {step}')
    nx.draw(network_data["graph"], network_data["pos"], node_color=network_data["colors"], labels=network_data["alpha_values"])
    plt.show()

# Plotting
plt.plot(range(1, model_steps + 1), model.pct_norm_abandonmnet, marker='o')
plt.xlabel('Step Number')
plt.ylabel('Percentage Adopt New Technology')
plt.title('Percentage of Agents Adopting New Technology Over Time')
plt.show()