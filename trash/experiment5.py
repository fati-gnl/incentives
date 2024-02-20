"""
Experiment5.py

This script conducts experiments to analyze the effect of different incentive distribution strategies and different total incentive
amounts on the adoption rates in a networked model.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from src.model import GameModel
from src.network_creation import create_connected_network
import multiprocessing as mp
from functools import partial
import time

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 20
Vh = 11
Vl = 8
p = 8
beta = 9999

network_type = "Erdos_Renyi"
entitled_distribution = "Uniform"
if network_type == "Erdos_Renyi":
    total_to_distribute_values = np.linspace(45000, 80000, 5)
elif network_type == "Barabasi":
    total_to_distribute_values = np.linspace(20000, 55000, 30)
elif network_type == "Homophily":
    total_to_distribute_values = np.linspace(25000, 80000, 40)

def run_experiment_for_strategy(incentive_strategy, G, total_to_distribute_values, size_network, random_seed, Vl, p, model_steps, beta):
    """
    Run experiments for a specific incentive strategy.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    :param int total_to_distribute_values: Amount of incentive to distribute
    :param int size_network: Number of nodes in the network.
    :param int random_seed: Seed for number generation.
    :param float Vl: Lowest reward for not selecting their preferred strategy.
    :param float p: Penalty for miscoordination with neighbours.
    :param int model_steps: Number of steps of the model to run
    :return: tuple (incentive_strategy, figures): A tuple containing the incentive strategy and its corresponding figures.
             Results include pairs of total incentive values and the final adoption percentages.
    """
    st = time.time()
    print("has started for " + str(incentive_strategy))

    # Results for the current strategy
    results = []

    # Run experiments for different total_to_distribute values
    for total_to_distribute in total_to_distribute_values:
        print(str(incentive_strategy) + " " + str(total_to_distribute))
        model = GameModel(
            num_agents=size_network, network=G, Vl=Vl, p=p,
            total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta
        )

        # Run the model
        model.step(max_steps = model_steps)

        # Collect figures
        results.append((total_to_distribute, model.pct_norm_abandonmnet[-1]))

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("has ended for " + str(incentive_strategy))

    # Return the figures for the current strategy
    return((incentive_strategy, results))

# Different incentive strategies
incentive_strategies = ["Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"]

# Results for each strategy
all_results = []

# Create connected network
G = create_connected_network(size_network, connectivity_prob, random_seed, Vh=Vh, gamma=True, type=network_type, entitled_distribution=entitled_distribution)

# Number of processes to use
num_processes = mp.cpu_count()

# Split incentive strategies into chunks
chunks = np.array([[strategy] for strategy in incentive_strategies])
print(chunks)

# Define a partial function to pass fixed arguments to run_experiment_for_strategy
partial_func = partial(
    run_experiment_for_strategy,
    G = G,
    total_to_distribute_values=total_to_distribute_values,
    size_network=size_network,
    random_seed=random_seed,
    Vl=Vl,
    p=p,
    model_steps=model_steps,
    beta=beta
)

# Run experiments for each chunk of incentive strategies in parallel
with mp.Pool(processes=num_processes) as pool:
    results = pool.map(partial_func, chunks)

# Combine the figures from all processes
all_results_combined = []

for result in results:
    incentive_strategy, results_list = result
    all_results_combined.append((incentive_strategy, results_list))

# Set markers per strategy
markers = ['s', '^', 'D', 'v', 'o']

print(all_results_combined)

for i, (incentive_strategy, results) in enumerate(all_results_combined):
    total_to_distribute_values, adoption_percentages = zip(*results)
    marker = markers[i % len(markers)]
    plt.plot(total_to_distribute_values, adoption_percentages, marker=marker, label=incentive_strategy[0])

# TODO: DIFFERENT FILE PATH PER COMBINATION
file_path = "minimum_incetives_per_strategy.txt"

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Incentive Strategy", "Total To Distribute Values", "Adoption Percentages"])
    for incentive_strategy, results_list in all_results_combined:
        for total_to_distribute, adoption_percentage in results_list:
            writer.writerow([incentive_strategy, total_to_distribute, adoption_percentage])


plt.xlabel('Incentive level')
#plt.xlabel('Incentive per Individual')
plt.ylabel('Final state, x(F)')
plt.ylim(-10, 110)
plt.tick_params(direction="in")
#plt.title('Equal incentive to all agents')
plt.legend()
plt.show()

# TODO: SAVE PLOT