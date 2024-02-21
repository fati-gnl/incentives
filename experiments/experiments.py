"""
Experiments.py

This script conducts experiments to analyze the effect of different incentive distribution strategies and different total incentive
amounts on the number of agents who receive an incentive.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from src.model import GameModel
from src.network_creation import create_connected_network
import multiprocessing as mp
from functools import partial
import time

# Constant parameters for all the experiments
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 20
Vh = 11
Vl = 8
p = 8
beta = 99

# Varies per experiment
entitled_distribution = "Uniform"
network_type = "Erdos_Renyi"

if network_type == "Erdos_Renyi":
    total_to_distribute_values = np.linspace(35000, 70000, 30)
elif network_type == "Barabasi":
    total_to_distribute_values = np.linspace(35000, 80000, 30)
elif network_type == "Homophily":
    total_to_distribute_values = np.linspace(25000, 80000, 40)

# Create connected network
G = create_connected_network(size_network, connectivity_prob, random_seed, Vh=Vh, gamma=True, type=network_type, entitled_distribution=entitled_distribution)

# Different incentive strategies
incentive_strategies = ["Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"]


def calculate_gini_coefficient(incentives):
    """
    Formula to calculate the Gini Coeffient given a list with the amount of incentives received by each agent.
    :param [floats] incentives: Incentive amount per agent
    """
    # Normalise the incentives
    total_incentives = np.sum(incentives)
    incentives = incentives / total_incentives

    diffsum = 0
    for i, xi in enumerate(incentives[:-1], 1):
        diffsum += np.sum(np.abs(xi - incentives[i:]))
    return diffsum / (len(incentives) ** 2 * np.mean(incentives))

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
    results_timesteps_95 = []
    results_spillovers = []
    results_incentivized = []
    results_norm = []
    results_gini = []

    # Run experiments for different total_to_distribute values
    for total_to_distribute in total_to_distribute_values:
        print(str(incentive_strategy) + " " + str(total_to_distribute))
        model = GameModel(
            num_agents=size_network, network=G, Vl=Vl, p=p,
            total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta
        )

        model.step(max_steps = model_steps)

        # Collect figures
        incentive_count = model.number_of_incentivised
        spillovers = model.spillovers
        inc_per_agent = model.total_incentives_ot

        gini_coeff = calculate_gini_coefficient(inc_per_agent)

        results_incentivized.append((total_to_distribute, incentive_count))
        results_timesteps_95.append((total_to_distribute, model.timesteps_95))
        results_spillovers.append((total_to_distribute, spillovers))
        results_norm.append((total_to_distribute, model.pct_norm_abandonmnet[-1]))
        results_gini.append((total_to_distribute, gini_coeff))


    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("has ended for " + str(incentive_strategy))

    # Return the figures for the current strategy
    return (incentive_strategy, results_timesteps_95, results_spillovers, results_incentivized, results_norm, results_gini)

# Number of processes to use
num_processes = mp.cpu_count()

# Split incentive strategies into chunks
chunks = np.array([[strategy] for strategy in incentive_strategies])

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

# Saving and combining the figures from all processes
results_tm = []
results_s = []
results_inc = []
results_count = []
results_gini = []

for result in results:
    incentive_strategy = result[0]
    timesteps_95, spillovers, incentivized, count, gini = result[1:]
    results_tm.append((incentive_strategy, timesteps_95))
    results_s.append((incentive_strategy, spillovers))
    results_inc.append((incentive_strategy, incentivized))
    results_count.append((incentive_strategy, count))
    results_gini.append((incentive_strategy, gini))


file_name = "results_{}_{}.txt".format(network_type, entitled_distribution)

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    result_data = [
        ("Time taken to achieve 95% adoption", results_tm),
        ("Number of spillovers", results_s),
        ("Number of agents incentivized", results_inc),
        ("Incentive count", results_count),
        ("Gini coefficient", results_gini)
    ]

    for header, result_list in result_data:
        writer.writerow(["Incentive Strategy", "Total to Distribute", header])
        for strategy, data in result_list:
            for total, value in data:
                writer.writerow([strategy, total, value])

def plot_and_save_results(results, filename, ylabel):
    """
    Plot and save the results of each experiment
    """
    markers = ['s', '^', 'D', 'v', 'o']
    plt.figure(figsize=(8, 6))
    plt.title(ylabel)
    for i, (incentive_strategy, results_data) in enumerate(results):
        total_to_distribute_values, data = zip(*results_data)
        marker = markers[i % len(markers)]
        plt.plot(total_to_distribute_values, data, marker=marker, label=incentive_strategy[0].replace("_", " ").strip("'[]'"))
    plt.xlabel('Incentive level')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig(filename)
    plt.close()

plot_and_save_results(results_tm,'{}_{}_time_to_95_adoption.png'.format(network_type, entitled_distribution), 'Time taken to achieve 95% adoption')
plot_and_save_results(results_s,'{}_{}_number_of_spillovers.png'.format(network_type, entitled_distribution), 'Number of spillovers')
plot_and_save_results(results_inc,'{}_{}_number_of_agents_incentivized.png'.format(network_type, entitled_distribution), 'Number of agents incentivized')
plot_and_save_results(results_count,'{}_{}_final_state.png'.format(network_type, entitled_distribution), 'Final state, x(F)')
plot_and_save_results(results_gini,'{}_{}_gini.png'.format(network_type, entitled_distribution), 'Gini Coefficient')