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

def run_experiment_for_strategy(incentive_strategy, total_to_distribute, num_runs, size_network, random_seeds, Vl, p,
                                model_steps, beta, amount_extra, G_list):
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
    random_seed = random_seeds[num_runs]

    G = G_list[num_runs]

    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra
    )

    game_model.step(max_steps=model_steps)

    # Collect metrics
    norm = game_model.pct_norm_abandonmnet[-1]

    return incentive_strategy, total_to_distribute, norm, num_runs


if __name__ == '__main__':

    start_t = time.time()
    print("Experiment started")

    entitled_distribution = "BiModal"
    network_types = ["Barabasi","Erdos_Renyi"]

    # Constant parameters for all the experiments
    size_network = 1000
    model_steps = 20
    num_runs = 30
    Vh = 11
    Vl = 8
    p = 8
    beta = 99999
    np.random.seed(123)
    amount_extra = 1
    random_seeds = [np.random.randint(10000) for _ in range(num_runs)]
    width_percentage = 0.675

    # Different incentive strategies
    incentive_strategies = ["Highest_gamma", "Complex_centrality", "Random", "Highest_degree", "Lowest_degree", "Lowest_gamma", "High_local_clustering_c", "Betweenness_centrality"]

    num_processes = mp.cpu_count()

    total_to_distribute_values = np.linspace(35000, 80000, 30)
    t_l = total_to_distribute_values.tolist()

    chunks = list(product(incentive_strategies, total_to_distribute_values, range(num_runs)))

    # Run experiments for each combination of entitled distribution, strategy, and incentive level
    results_plot = []
    results_q1 = []
    results_q3 = []
    for y, network_type in enumerate(network_types):

        def create_network_for_seed(size_network, random_seed, Vh, network_type, entitled_distribution, width_percentage, Vl):
            return create_connected_network(size_network, random_seed, Vh=Vh, type=network_type,
                                            entitled_distribution=entitled_distribution, width_percentage = width_percentage, Vl = Vl)


        with mp.Pool(processes=num_processes) as pool:
            G_list = pool.starmap(create_network_for_seed,
                                  [(size_network, seed, Vh, network_type, entitled_distribution, width_percentage, Vl) for
                                   seed in random_seeds])

        # Run experiments for each combination of strategy and incentive level
        partial_func = partial(run_experiment_for_strategy, size_network=size_network,
                                random_seeds=random_seeds, Vl=Vl,
                                p=p, model_steps=model_steps, beta=beta, amount_extra=amount_extra, G_list= G_list)

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(partial_func, chunks)

        # Store results
        first_passage_indices = [[[] for x in total_to_distribute_values] for _ in incentive_strategies]
        for i, result in enumerate(results):
            strategy, total_d, count, num_runs = result
            print("num_runs", num_runs)
            ind = t_l.index(total_d)
            idx_str = incentive_strategies.index(strategy)
            print("first_passage_indices[idx_str][ind]", first_passage_indices[idx_str][ind])
            first_passage_indices[idx_str][ind].append(count)

        indexes = []
        for strategy in first_passage_indices:
            transposed = np.array(strategy).T
            idx_str = []
            for list in transposed:
                idx = np.where(list == 1)[0]
                idx_str.append(idx[0])
            indexes.append(idx_str)

        print("indexes", indexes)

        averages_indices = [sum(sublist) / len(sublist) for sublist in indexes]
        print("averages_indices", averages_indices)

        q1 = [round(np.percentile(sublist, 25)) for sublist in indexes]
        q3 = [round(np.percentile(sublist, 75)) for sublist in indexes]

        results_q1.append(q1)
        results_q3.append(q3)
        results_plot.append(averages_indices)

    print(results_plot)

    sorted_indices = []

    def interpolate_value(array, index):
        lower_index = int(index)
        upper_index = lower_index + 1

        if upper_index >= len(array):
            return array[-1]

        lower_value = array[lower_index]
        upper_value = array[upper_index]

        fractional_part = index - lower_index

        interpolated_value = lower_value + fractional_part * (upper_value - lower_value)
        return interpolated_value

    # Plotting
    markers = ['s', '^', 'D', 'v', 'o', '*', 'x', '+', '.']

    modified_labels = [strategy.replace("_", " ").strip("'[]'") for strategy in incentive_strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(results_plot):
        if i == 0:
            sorted_indices.append(np.argsort([interpolate_value(total_to_distribute_values, result[i]) for i in range(len(result))]))
        print("sorted_indices", sorted_indices)
        sorted_indices_concat = np.concatenate(sorted_indices)
        result_ordered = [interpolate_value(total_to_distribute_values, result[i]) for i in sorted_indices_concat]
        print("result_ordered", result_ordered)
        results_q1f = [interpolate_value(total_to_distribute_values, results_q1[i][j]) for j in sorted_indices_concat]
        print("results_q1f", results_q1f)
        results_q3f = [interpolate_value(total_to_distribute_values, results_q3[i][j]) for j in sorted_indices_concat]

        ax.plot(incentive_strategies, result_ordered, marker=markers[i], label=network_types[i].replace("_", " ").strip("'[]'"))
        ax.fill_between(incentive_strategies, results_q1f, results_q3f, alpha=0.2)

    plt.ylabel('Incentive Level')
    #plt.title('Minimum incentive amount for the first passage of time')
    plt.xticks(range(len(incentive_strategies)), [modified_labels[idx] for idx in np.concatenate(sorted_indices)])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig('res_network_final_{}_new88.png'.format(network_type))
    plt.close()

    end_t = time.time()
    elapsed_time = end_t - start_t
    print(f"Execution time: {elapsed_time} seconds")
    print("Experiment ended")