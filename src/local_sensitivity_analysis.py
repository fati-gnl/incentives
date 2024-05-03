"""
global_sensitivity_analysis.py

In this file, a local sensitivy analysis has been conducted to assess the potential
for complete abandonment of a detrimental norm within a network featuring
heterogeneous individuals.
"""
import multiprocessing as mp
import warnings
import numpy as np
from SALib.sample import sobol
from SALib.analyze import sobol as sb_analyze
from matplotlib import pyplot as plt
from itertools import product
from functools import partial

import src.agent_sorting as agent_sorting
from src.model import GameModel
from src.network_creation import create_connected_network

network_type = "Homophily"
entitled_distribution = "Normal"
incentive_strategies = ["Highest_gamma", "Random", "Lowest_gamma"]

warnings.filterwarnings("ignore", category=RuntimeWarning)

random_seed = 123
beta = 99999
size_network = 1000
num_runs = 20
amount_extra = 1
random_seeds = [np.random.randint(10000) for _ in range(num_runs)]
total_to_distribute_values = np.linspace(1000, 200000, 40)

problem = {
    'num_vars': 4,
    'names': ['Average Degree', 'Vh', 'Width percentage', 'P_in'],
    'bounds': [[10, 50],
               [9, 15],
               [0.1, 1],
               [0, 1]]
}

param_values = sobol.sample(problem, 600, calc_second_order=False)

def run_experiment_for_strategy(num_runs, param_value, incentive_strategy,
                                total_to_distribute_values, size_network, amount_extra,
                                random_seeds):
    """
    This function determines for a given parameter set whether or not the complete abandonment of a detrimental
    norm can be achieved.
    :param num_runs: Number of simulation runs
    :param param_value: Specific combination of parameter values to perform SA on: 'Average Degree', 'Vh', 'Width percentage', 'P_in'.
    :param incentive_strategy: Current incentive distribution strategy
    :param total_to_distribute_values: Range of total incentive amounts to consider
    :param size_network: Size of the network
    :param float amount_extra: Extra amount of money that should be given to individuals to make the option of changing strategies more favorable.
    :param random_seeds: Random seed used per simulation run
    :return: num_runs, param_value, incentive_strategy, found_a_value
    """

    average_degree, Vh, width_percentage, p_in = param_value

    beta = 9999
    Vl = 8
    p = 8

    random_seed = random_seeds[num_runs]

    G = create_connected_network(size_network, random_seed, Vh=Vh, type=network_type,
                                        entitled_distribution=entitled_distribution, average_degree=average_degree, width_percentage=width_percentage, Vl=Vl, p_in=p_in)

    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    total_to_distribute_value = total_to_distribute_values[-1]

    game_model = GameModel(
        num_agents=int(size_network), network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute_value, seed=random_seed, incentive_strategy=incentive_strategy,
        beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra
    )
    game_model.step(max_steps=25)

    # Calculate whether the simulation converged or not for this specific set of parameter values
    if game_model.pct_norm_abandonmnet[-1] == 1:
        found_a_value = 1
    else:
        found_a_value = 0

    return num_runs, param_value, incentive_strategy, found_a_value


if __name__ == '__main__':

    num_processes = mp.cpu_count()

    partial_func = partial(
        run_experiment_for_strategy,
        total_to_distribute_values=total_to_distribute_values,
        size_network=size_network,
        amount_extra=amount_extra,
        random_seeds=random_seeds
    )

    chunks = list(product(range(num_runs), param_values, incentive_strategies))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(partial_func, chunks)

    Y = [[[] for _ in range(len(param_values))] for _ in range(len(incentive_strategies))]
    for result in results:
        num_runs, param_value, incentive_strategy, found_a_value = result

        ind_incentive_strategy = incentive_strategies.index(incentive_strategy)
        ind_param_value = np.where((param_values == param_value).all(axis=1))[0][0]

        Y[ind_incentive_strategy][ind_param_value].append(found_a_value)

    print("Y", np.array(Y))

    # Aggregates the results per incentive strategy for the different simulation runs by using the mean
    Y1 = [[np.median(sublist) for sublist in sublist_list] for sublist_list in Y]
    Y1_array = np.array(Y1)

    print("Y1", Y1_array)

    ranked_Y = [sorted(sublist) for sublist in Y1_array.T]
    print("ranked_Y", ranked_Y)

    Y_sum = np.sum(np.array(ranked_Y).T, axis=0)
    print("Y_sum", Y_sum)

    # The results were then aggregated across different incentive strategies for each parameter value to mitigate intervention variability.
    # Each parameter combination’s highest gamma strategy was then divided by the sum of all other incentive strategies for that specific parameter value.
    Y_final = np.divide(Y1_array[0], Y_sum, out=np.zeros_like(Y_sum), where=Y_sum != 0)

    results = sb_analyze.analyze(problem, Y_final, calc_second_order=False, print_to_console=False)

    print("results", results)

    # Sobol indices
    plt.figure(figsize=(10, 6))
    results.plot()
    plt.tight_layout()
    plt.savefig("sobol_sa_first_plot_new.png")
    plt.close()