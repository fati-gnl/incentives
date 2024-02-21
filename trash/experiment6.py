"""
Experiment6.py
Distribution of incentives
"""
import matplotlib.pyplot as plt
import numpy as np
from src.model import GameModel
from src.network_creation import create_connected_network
import multiprocessing as mp
from functools import partial
import time
from collections import Counter

def run_experiment_for_strategy(incentive_strategy, G, total_to_distribute, size_network, random_seed, Vl, p, model_steps):
    """
    Run experiments for a specific incentive strategy.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    :param int total_to_distribute_value: Amount of incentive to distribute
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

    # Run experiments for different total_to_distribute values
    model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy
    )

    # Run the model
    for step in range(model_steps):
        model.step()

    # Collect figures
    incentive_distribution = np.round(model.datacollector.get_agent_vars_dataframe()["Incentive"].values)
    incentive_counter = Counter(incentive_distribution)

    sorted_incentives = sorted(incentive_counter.keys())
    frequencies = [incentive_counter[incentive] for incentive in sorted_incentives]

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("has ended for " + str(incentive_strategy))

    # Return the figures for the current strategy
    return((incentive_strategy, (sorted_incentives, frequencies)))


# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
p = 8

# Different incentive strategies
incentive_strategies = ["Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"]

total_to_distribute = 25000

# Results for each strategy
all_results = []

# Create connected network
G = create_connected_network(size_network, connectivity_prob, random_seed, Vh=Vh, gamma=True, type="Erdos_Renyi")

# Number of processes to use
num_processes = mp.cpu_count()

# Split incentive strategies into chunks
chunks = np.array([[strategy] for strategy in incentive_strategies])
print(chunks)

# Define a partial function to pass fixed arguments to run_experiment_for_strategy
partial_func = partial(
    run_experiment_for_strategy,
    G = G,
    total_to_distribute=total_to_distribute,
    size_network=size_network,
    random_seed=random_seed,
    Vl=Vl,
    p=p,
    model_steps=model_steps
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
    print(f"Strategy: {incentive_strategy}, Results: {results}")
    total_to_distribute_values, incentive_count_percentages = results
    marker = markers[i % len(markers)]
    plt.bar(total_to_distribute_values, incentive_count_percentages, label=incentive_strategy[0], alpha=0.5)


plt.xlabel('Incentive amount')
plt.ylabel('Number of agents incentivised')
plt.tick_params(direction="in")
plt.legend()
plt.show()