"""
Experiment1.py

Adoption of New Technology Over Time for Different Count Numbers

This experiment investigates the adoption of new technology over time for varying count numbers.
Different parameters such as network structure, gamma distribution, and initiator strategy can be adjusted.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.model import GameModel
from src.network_creation import create_connected_network

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
p = 8
betas = [1000,100,10,1,0.1,0.001]

# Different values for total_to_distribute
#total_to_distribute_values = np.linspace(75000, 75000, 1)

# Initialize list to store figures
results = []

np.random.seed(random_seed)
G = create_connected_network(size_network, connectivity_prob, random_seed, Vh=Vh, gamma=True, type="Erdos_Renyi", entitled_distribution="Uniform")

# Loop through different initiators values
#for incentive_amount in total_to_distribute_values:
incentive_amount = 70000
for beta in betas:

    # Generate a connected no_gamma network
    model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=incentive_amount, seed=random_seed, incentive_strategy="Highest_gamma", beta=beta
    )

    # Run the model
    model.step(max_steps=model_steps)

    # Save the percentage of agents adopting new technology for each step
    #results.append((incentive_amount, model.pct_norm_abandonmnet))
    #results.append((incentive_amount, model.incentive_amounts))
    results.append((beta, (model.sigmoid_inputs, model.transition_probs)))


for beta, (sigmoid_inputs, transition_probs) in results:
    plt.plot(sigmoid_inputs, transition_probs, 'o', label=f'Beta = {beta}')

# Add labels and legend
plt.xlabel('Sigmoid Input')
plt.ylabel('Transition Probability')
plt.title('Effect of Beta on Transition Probability')
plt.legend()
plt.grid(True)
# Show the plot
plt.show()


# Plotting
#for incentive_amount, pct_values in results:
    #plt.plot(range(1, model_steps + 1), pct_values, marker='o', label=f"{incentive_amount}")
    #plt.plot(range(1, len(pct_values) + 1), pct_values, marker='o', label=f"{incentive_amount}")

#plt.xlabel("Step Number")
#plt.ylabel("Percentage of Adoption")
#plt.ylabel("Amount given to an agent")
#plt.tick_params(direction="in")
#plt.legend()
#plt.show()
