"""
Experiments.py

This script runs all of the different experiments built to analyze the effect of different incentive amounts on various factors.
In addition, it produces the figures corresponding to each result:
Experiments:
1. Time taken to achieve 95% adoption
2. Number of spillovers
3. Number of agents who received an incentive
4. Proportion of agents who transitioned
5. Gini Coefficient
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from src.model import GameModel
from src.network_creation import create_connected_network
import multiprocessing as mp
from functools import partial
import time

st = time.time()
print("Experiment started")

# Constant parameters for all the experiments
size_network = 1000
connectivity_prob = 0.05
model_steps = 20
Vh = 11
Vl = 8
p = 8
beta = 99
num_runs = 5
np.random.seed(123)
random_seeds = [np.random.randint(10000) for _ in range(num_runs)]

# Varies per experiment
entitled_distribution = "Normal"
network_type = "Erdos_Renyi"

if network_type == "Erdos_Renyi":
    total_to_distribute_values = np.linspace(35000, 70000, 2)
elif network_type == "Barabasi":
    total_to_distribute_values = np.linspace(35000, 80000, 30)
elif network_type == "Homophily":
    total_to_distribute_values = np.linspace(25000, 80000, 40)

# Create connected network
G = create_connected_network(size_network, connectivity_prob, 123, Vh=Vh, gamma=True, type=network_type, entitled_distribution=entitled_distribution)

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
    :return:
    """
    results_timesteps_95 = []
    results_spillovers = []
    results_incentivized = []
    results_norm = []
    results_gini = []

    # Run experiments for different total_to_distribute values
    for total_to_distribute in total_to_distribute_values:
        model = GameModel(
            num_agents=size_network, network=G, Vl=Vl, p=p,
            total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta
        )
        model.step(max_steps = model_steps)

        # Collect figures
        spillovers = model.spillovers
        inc_per_agent = model.total_incentives_ot
        gini_coeff = calculate_gini_coefficient(inc_per_agent)

        print("Received incentive but did not transition count: " + str(model.inc_but_no_transition)) if model.inc_but_no_transition != 0 else None

        results_incentivized.append(sum(1 for amount in inc_per_agent if amount != 0))
        results_timesteps_95.append(model.timesteps_95 if model.has_reached_95 else 0)
        results_spillovers.append(spillovers)
        results_norm.append(model.pct_norm_abandonmnet[-1])
        results_gini.append(gini_coeff)

    # Return the figures for the current strategy
    return (incentive_strategy, results_timesteps_95, results_spillovers, results_incentivized, results_norm, results_gini)

num_processes = mp.cpu_count()
chunks = np.array([[strategy] for strategy in incentive_strategies])

results_all_sim = {'tm': [],'s': [],'inc': [],'count': [],'gini': []}

# Run experiments for each chunk of incentive strategies in parallel
for i in range(num_runs):

    partial_func = partial(
        run_experiment_for_strategy,
        G=G,
        total_to_distribute_values=total_to_distribute_values,
        size_network=size_network,
        random_seed= random_seeds[i],
        Vl=Vl,
        p=p,
        model_steps=model_steps,
        beta=beta
    )
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(partial_func, chunks)

    for i, result in enumerate(results):
        incentive_strategy = result[0]
        timesteps_95, spillovers, incentivized, count, gini = result[1:]

        results_all_sim['tm'].append((incentive_strategy, timesteps_95))
        results_all_sim['s'].append((incentive_strategy, spillovers))
        results_all_sim['inc'].append((incentive_strategy, incentivized))
        results_all_sim['count'].append((incentive_strategy, count))
        results_all_sim['gini'].append((incentive_strategy, gini))

def calculate_average_values(data):
    """
    Calculate the average values for each incentive level across multiple runs.
    """
    strategy_names = ["Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"]
    strategy_data = [[] for _ in range(len(strategy_names))]

    for simulation in data:
        strategy_name = simulation[0][0]
        index = strategy_names.index(strategy_name)
        strategy_data[index].extend([simulation[1]])

    averaged_data = []
    for values in strategy_data:
        averaged_data.append([sum(x) / len(x) for x in zip(*values)])

    return averaged_data

# Calculate average values for each metric
average_results = {}
for metric, results in results_all_sim.items():
    average_results[metric] = calculate_average_values(results)

file_name = "results_{}_{}.txt".format(network_type, entitled_distribution)

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["Metric", "Incentive Strategy", "Average Value"]
    writer.writerow(header)

    for metric, results in average_results.items():
        for i, strategy_data in enumerate(results):
            strategy_name = incentive_strategies[i]
            for value in strategy_data:
                writer.writerow([metric, strategy_name, value])

def plot_and_save_results(results, filename, xdata, ylabel, incentive_strategies):
    """
    Plot and save the results of each experiment
    """
    markers = ['s', '^', 'D', 'v', 'o']
    plt.figure(figsize=(8, 6))
    plt.title(ylabel)
    for i, data in enumerate(results):
        marker = markers[i % len(markers)]
        plt.plot(xdata, data, marker=marker, label=incentive_strategies[i].replace("_", " ").strip("'[]'"))
    plt.xlabel('Incentive level')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig(filename)
    plt.close()

ylabels =['Time taken to achieve 95% adoption', 'Number of spillovers', 'Number of agents incentivized', 'Final state, x(F)', 'Gini Coefficient']
file_titles = ['time_to_95_adoption', 'number_of_spillovers', 'number_of_agents_incentivized', 'final_state', 'gini']

for i, (metric, results) in enumerate(average_results.items()):
    plot_and_save_results(results, '{}_{}_{}3.png'.format(network_type, entitled_distribution, file_titles[i]), total_to_distribute_values, ylabels[i], incentive_strategies)

et = time.time()
elapsed_time = et - st
print(f"Execution time for {incentive_strategy}: {elapsed_time} seconds")
print("Experiment ended for:", incentive_strategy)
