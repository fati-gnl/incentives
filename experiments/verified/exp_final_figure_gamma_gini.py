import time
import src.agent_sorting as agent_sorting

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from functools import partial
from itertools import product
import warnings

from src.model import GameModel
from src.network_creation import create_connected_network

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def run_experiment_for_strategy(num_rum, incentive_strategy, size_network, random_seeds, Vl, p,
                                model_steps, beta, amount_extra, total_to_distribute, width_percentage, entitled_distribution, Vh, network_type):
    """
    Run experiments for a specific incentive strategy.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    :param int total_to_distribute: Amount of incentive to distribute
    :param nx.Graph G: Graph representing the network.
    :param int size_network: Number of nodes in the network.
    :param int random_seed: Seed for number generation.
    :param float Vl: Lowest reward for not selecting their preferred strategy.
    :param float p: Penalty for miscoordination with neighbours.
    :param int model_steps: Number of steps of the model to run
    :param float amount_extra: Extra amount of money that should be given to individuals to make the option of changing strategies more favorable.
    :return:
    """

    random_seed = random_seeds[num_rum]

    G = create_connected_network(size_network, random_seed, Vh=Vh,
                                 type=network_type, entitled_distribution=entitled_distribution,
                                 width_percentage=width_percentage, Vl=Vl)

    # Sort by incentive distribution for each strategy
    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra
    )
    game_model.step(max_steps=model_steps)

    # Collect metrics
    inc_per_agent = game_model.total_incentives_ot
    gini_coeff = round(calculate_gini_coefficient(inc_per_agent),2)

    return incentive_strategy, num_rum, gini_coeff

def run_experiment_for_strategy2(num_rum, size_network, random_seeds, Vl, p,
                                model_steps, beta, amount_extra, total_to_distribute, G_list, sort_by_incentive_dist_l, incentive_strategy):
    """
    Run experiments for a specific incentive strategy.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    :param int total_to_distribute: Amount of incentive to distribute
    :param nx.Graph G: Graph representing the network.
    :param int size_network: Number of nodes in the network.
    :param int random_seed: Seed for number generation.
    :param float Vl: Lowest reward for not selecting their preferred strategy.
    :param float p: Penalty for miscoordination with neighbours.
    :param int model_steps: Number of steps of the model to run
    :param float amount_extra: Extra amount of money that should be given to individuals to make the option of changing strategies more favorable.
    :return:
    """

    random_seed = random_seeds[num_rum]
    sort_by_incentive_dist = sort_by_incentive_dist_l[num_rum]

    G = G_list[num_rum]

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra
    )
    game_model.step(max_steps=model_steps)

    # Collect metrics
    inc_per_agent = game_model.total_incentives_ot
    gini_coeff = round(calculate_gini_coefficient(inc_per_agent),2)

    return incentive_strategy, num_rum, gini_coeff


if __name__ == '__main__':

    start_t = time.time()
    print("Experiment started")

    entitled_distributions = ["Normal", "Uniform", "BiModal"]
    network_type = "Barabasi"

    # Constant parameters for all the experiments
    size_network = 1000
    model_steps = 20
    Vh = 11
    Vl = 8
    p = 8
    beta = 99999
    num_runs = 30
    np.random.seed(123)
    amount_extra = 1
    random_seeds = [np.random.randint(10000) for _ in range(num_runs)]

    width_percentage = 0.675

    total_to_distribute = 80000

    # Different incentive strategies
    incentive_strategies = ["Highest_gamma", "Complex_centrality", "Random", "Highest_degree", "Lowest_degree"
                            , "Lowest_gamma", "High_local_clustering_c", "Betweenness_centrality"]

    modified_labels = [strategy.replace("_", " ").strip("'[]'") for strategy in incentive_strategies]

    # Run experiments for each combination of entitled distribution, strategy, and incentive level
    results_plot = []
    results_q1 = []
    results_q3 = []

    for entitled_distribution in entitled_distributions:

        num_processes = mp.cpu_count()
        chunks = list(product(range(num_runs), incentive_strategies[:-1]))

        # Run experiments for each combination of strategy and incentive level
        partial_func = partial(run_experiment_for_strategy, size_network=size_network, random_seeds=random_seeds, Vl=Vl,
                                   p=p, model_steps=model_steps, beta=beta,
                                   amount_extra=amount_extra, total_to_distribute=total_to_distribute,
                                   width_percentage=width_percentage, entitled_distribution=entitled_distribution, Vh=Vh, network_type=network_type)

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(partial_func, chunks)

        # Repeat the process just for betweeness centrality
        def create_network_for_seed(size_network, random_seed, Vh, network_type, entitled_distribution,
                                    width_percentage, Vl):
            return create_connected_network(size_network, random_seed, Vh=Vh, type=network_type,
                                            entitled_distribution=entitled_distribution,
                                            width_percentage=width_percentage, Vl=Vl, average_degree=50)

        with mp.Pool(processes=num_processes) as pool:
            G_list = pool.starmap(create_network_for_seed,
                                  [(size_network, seed, Vh, network_type, entitled_distribution, width_percentage, Vl)
                                   for
                                   seed in random_seeds])

        chunks2 = list(product(range(num_runs)))

        sort_by_incentive_dist_t = []
        for x in range(num_runs):
            sort_by_incentive_dist_t.append(agent_sorting.sort_by_incentive_dist(G_list[x], random_seeds[x], incentive_strategies[-1]))

        partial_func2 = partial(run_experiment_for_strategy2, size_network=size_network, random_seeds=random_seeds, Vl=Vl,
                               p=p, model_steps=model_steps, beta=beta,
                               amount_extra=amount_extra, total_to_distribute=total_to_distribute,
                               G_list=G_list, sort_by_incentive_dist_l=sort_by_incentive_dist_t, incentive_strategy=incentive_strategies[-1])

        with mp.Pool(processes=num_processes) as pool:
            results2 = pool.starmap(partial_func2, chunks2)

        results_final = [[] for _ in incentive_strategies]
        for i, result in enumerate(results):
            incentive_strategy, num_rum, gini_coeff = result

            idx_str = incentive_strategies.index(incentive_strategy)
            results_final[idx_str].append(gini_coeff)

        for i, result in enumerate(results2):
            incentive_strategy, num_rum, gini_coeff = result

            idx_str = incentive_strategies.index(incentive_strategy)
            results_final[idx_str].append(gini_coeff)

        averages_indices = [sum(sublist) / len(sublist) for sublist in results_final]

        q1 = [np.percentile(sublist, 25) for sublist in results_final]
        q3 = [np.percentile(sublist, 75) for sublist in results_final]

        print("averages_indices", averages_indices)
        print("q1", q1)
        print("q3", q3)

        results_q1.append(q1)
        results_q3.append(q3)
        results_plot.append(averages_indices)

    print("results_plot", results_plot)

    sorted_indices = []

    # Plotting
    markers = ['s', '^', 'D', 'v', 'o', '*', 'x', '+', '.']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(results_plot):
        if i == 0:
            sorted_indices.append(np.argsort(results_plot[0]))
        print("sorted_indices", sorted_indices)
        sorted_indices_concat = np.concatenate(sorted_indices)
        print(sorted_indices_concat, "sorted_indices_concat")
        print(result, "result")
        result_ordered = [result[j] for j in sorted_indices_concat]
        results_q1f = [results_q1[i][j] for j in sorted_indices_concat]
        results_q3f = [results_q3[i][j] for j in sorted_indices_concat]

        ax.plot(incentive_strategies, result_ordered, marker=markers[i], label=entitled_distributions[i].replace("_", " ").strip("'[]'"))
        ax.fill_between(incentive_strategies, results_q1f, results_q3f, alpha=0.2)

    plt.ylabel('Gini Coefficient')
    plt.xticks(range(len(incentive_strategies)), [modified_labels[idx] for idx in np.concatenate(sorted_indices)])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig('exp_gamma_final_gini_{}.png'.format(network_type))
    plt.close()

    end_t = time.time()
    elapsed_time = end_t - start_t
    print(f"Execution time: {elapsed_time} seconds")
    print("Experiment ended")