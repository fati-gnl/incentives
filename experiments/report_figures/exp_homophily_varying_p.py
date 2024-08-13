"""
Exp_homophily_varying_p.py
In this file, a plot gets created that represents the average norm abandonment given a maximum amount to distribute.
The x axis is therefore the average norm abandonment of a population and the y the maximum amount to distribute.
This has been plotted for different p_in values.
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from functools import partial
from src.model import GameModel
from src.network_creation import create_connected_network
import src.agent_sorting as agent_sorting
import warnings
from itertools import product

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_experiment_for_strategy(run_num, p_in, total_to_distribute, size_network, random_seeds, Vl, p,
                                model_steps, beta, incentive_strategy, amount_extra, Vh, network_type, entitled_distribution, width_percentage):


    random_seed = random_seeds[run_num]

    G = create_connected_network(size_network, random_seed, Vh=Vh, type=network_type,width_percentage = width_percentage, Vl = Vl,
                                 entitled_distribution=entitled_distribution, p_in=p_in)

    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra)

    game_model.step(max_steps=model_steps)

    norm = game_model.pct_norm_abandonmnet[-1]

    return run_num, total_to_distribute, norm, p_in


if __name__ == '__main__':
    start_t = time.time()
    print("Experiment started")

    incentive_strategy = "Highest_gamma"
    network_type = "Homophily"
    entitled_distribution = "Normal"

    size_network = 1000
    connectivity_prob = 0.05
    model_steps = 20
    Vh = 11
    Vl = 8
    p = 8
    beta = 99999
    num_runs = 20
    np.random.seed(123)
    amount_extra = 1
    random_seeds = [np.random.randint(10000) for _ in range(num_runs)]

    width_percentage = 0.675

    total_to_distribute_values = np.linspace(25000, 80000, 40)
    t_l = total_to_distribute_values.tolist()

    p_ins = np.round(np.linspace(0, 1, 6), 2)

    num_processes = mp.cpu_count()
    chunks = list(product(range(num_runs), p_ins, total_to_distribute_values))

    partial_func = partial(run_experiment_for_strategy, size_network=size_network,
            random_seeds=random_seeds, Vl=Vl, p=p, model_steps=model_steps, beta=beta,
            incentive_strategy = incentive_strategy, amount_extra=amount_extra, Vh= Vh,
            network_type = network_type, entitled_distribution = entitled_distribution, width_percentage=width_percentage)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(partial_func, chunks)

    results_final = []
    results_final = [[[] for _ in p_ins] for _ in total_to_distribute_values]
    for result in results:
        run_num, total_to_distribute, norm, p_in = result
        index = t_l.index(total_to_distribute)
        p_in_ind = np.where(p_ins == p_in)[0][0]
        results_final[index][p_in_ind].append(norm)

    print("results_final", results_final)
    results_means = [[np.median(sublist) for sublist in sublist_list] for sublist_list in results_final]
    results_q1 = [[np.percentile(sublist, 25) for sublist in list] for list in results_final]
    results_q3 = [[np.percentile(sublist, 75) for sublist in list] for list in results_final]
    print("results_final1", results_means)

    results_means = np.array(results_means).T
    results_q1 = np.array(results_q1).T
    results_q3 = np.array(results_q3).T

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(results_means):
        ax.plot(result, total_to_distribute_values, marker='o', label=f'p_in={p_ins[i]}')
        ax.fill_betweenx(total_to_distribute_values, results_q1[i], results_q3[i], alpha=0.2)

    plt.xlabel('Average norm abandonment (Final state x(F)))')
    plt.ylabel('Maximum total amount to distribute')
    plt.title('Average norm abandonment given a maximum amount to distribute in a {} network with {} distribution and {} strategy'.format(network_type, entitled_distribution, incentive_strategy))
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig("Plot_homophily_{}_{}_new.png".format(entitled_distribution, incentive_strategy))
    plt.close()

    end_t = time.time()
    elapsed_time = end_t - start_t
    print(f"Execution time: {elapsed_time} seconds")
    print("Experiment ended")
