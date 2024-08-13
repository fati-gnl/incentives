"""
Exp_homophily_minimum_inc.py
This file makes a plot where the x axis are p_in values and the y axis represent the minimum total amount
to distribute required for a 95 % norm abandonment.
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from itertools import product
from src.model import GameModel
from src.network_creation import create_connected_network
import src.agent_sorting as agent_sorting
import warnings
from functools import partial

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_experiment_for_strategy(run_num, p_in, incentive_strategy, size_network, random_seeds, Vl, p,
                                model_steps, beta, amount_extra, target_norm, Vh, network_type, entitled_distribution, width_percentage):

    random_seed = random_seeds[run_num]

    G = create_connected_network(size_network, random_seed, Vh=Vh, type=network_type, width_percentage = width_percentage, Vl = Vl,
                                 entitled_distribution=entitled_distribution, p_in=p_in)

    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    total_to_distribute = 300000

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
        sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra)

    game_model.step(max_steps=model_steps)

    norm = game_model.pct_norm_abandonmnet[-1]

    if norm == 1:
        total_to_distribute = 10000
        while True:
            game_model = GameModel(
                num_agents=size_network, network=G, Vl=Vl, p=p,
                total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta,
                sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra)

            game_model.step(max_steps=model_steps)

            norm = game_model.pct_norm_abandonmnet[-1]

            if norm >= target_norm:
                break

            total_to_distribute += 1000

    return run_num, total_to_distribute, incentive_strategy


if __name__ == '__main__':
    start_t = time.time()
    print("Experiment started")

    entitled_distribution = "Normal"
    network_type = "Homophily"

    size_network = 1000
    model_steps = 20
    Vh = 11
    Vl = 8
    p = 8
    beta = 99999
    num_runs = 20
    np.random.seed(123)
    amount_extra = 1
    random_seeds = [np.random.randint(10000) for _ in range(num_runs)]
    p_ins = np.round(np.linspace(0, 1, 30), 2)

    width_percentage = 0.675

    target_norm = 0.95

    num_processes = mp.cpu_count()

    incentive_strategies = ["Highest_gamma", "Random", "Lowest_gamma"]

    chunks = list(product(range(num_runs), p_ins, incentive_strategies))

    partial_func = partial(run_experiment_for_strategy, size_network=size_network,
        random_seeds=random_seeds, Vl=Vl, p=p, model_steps=model_steps, beta=beta, amount_extra=amount_extra,
        target_norm=target_norm, Vh = Vh, network_type = network_type, entitled_distribution = entitled_distribution, width_percentage= width_percentage)

    results_final = [[[] for _ in range(num_runs)] for _ in incentive_strategies]
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(partial_func, chunks)

    for result in results:
        run_num, total_to_distribute, incentive_strategy = result
        index = incentive_strategies.index(incentive_strategy)
        results_final[index][run_num].append(total_to_distribute)

    averaged_results = []
    q1 = []
    q3 = []

    for strategy_results in results_final:
        transposed_results = np.array(strategy_results).T
        averaged_result = np.median(transposed_results, axis=1)
        q1.append(np.percentile(transposed_results, 5, axis=1))
        q3.append(np.percentile(transposed_results, 95, axis=1))
        averaged_results.append(averaged_result)

    print("averaged_results1", averaged_results)
    print("q1", q1)
    print("q3", q3)



    # Erdos - Barabasi

    def run_experiment_for_strategy2(run_num, incentive_strategy, network_type, size_network, random_seeds, Vl, p,
                                    model_steps, beta, amount_extra, target_norm, Vh,
                                    entitled_distribution, width_percentage):

        print("has entered")
        random_seed = random_seeds[run_num]

        G = create_connected_network(size_network, random_seed, Vh=Vh, type=network_type, width_percentage = width_percentage, Vl = Vl,
                                     entitled_distribution=entitled_distribution, p_in=0)

        sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

        total_to_distribute = 10000
        while True:
            game_model = GameModel(
                num_agents=size_network, network=G, Vl=Vl, p=p,
                total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy,
                beta=beta,
                sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra)

            game_model.step(max_steps=model_steps)

            norm = game_model.pct_norm_abandonmnet[-1]

            if norm >= target_norm:
                break

            total_to_distribute += 1000

        return run_num, total_to_distribute, incentive_strategy, network_type

    network_types = ["Erdos_Renyi", "Barabasi"]
    chunks1 = list(product(range(num_runs), incentive_strategies, network_types))

    partial_func1 = partial(run_experiment_for_strategy2, size_network=size_network,
                           random_seeds=random_seeds, Vl=Vl, p=p, model_steps=model_steps, beta=beta,
                           amount_extra=amount_extra,
                           target_norm=target_norm, Vh=Vh,
                           entitled_distribution=entitled_distribution, width_percentage=width_percentage)

    with mp.Pool(processes=num_processes) as pool:
        results2 = pool.starmap(partial_func1, chunks1)

    results_e_b = [[[] for _ in incentive_strategies] for _ in network_types]
    for result in results2:
        run_num, total_to_distribute, incentive_strategy, network_type = result
        inx = network_types.index(network_type)
        index = incentive_strategies.index(incentive_strategy)
        results_e_b[inx][index].append(total_to_distribute)

    print("results_e_b", results_e_b)

    results_means = [[sum(sublist) / len(sublist) for sublist in list] for list in results_e_b]
    results_q1 = [[np.percentile(sublist, 5) for sublist in list] for list in results_e_b]
    results_q3 = [[np.percentile(sublist, 95) for sublist in list] for list in results_e_b]


    markers = ['s', '^', 'D', 'v', 'o', '*', 'x', '+', '.']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, strategy_results in enumerate(averaged_results):
        indices_to_exclude = [idx for idx, total_to_distribute in enumerate(strategy_results) if
                              total_to_distribute == 300000]

        filtered_results = [value for idx, value in enumerate(strategy_results) if idx not in indices_to_exclude]
        filtered_p_ins = [p_ins[idx] for idx in range(len(p_ins)) if idx not in indices_to_exclude]
        print("q1", q1)
        print("q3", q3)
        print("q3[i]", q3[i])
        q1_f = [q1[i][idx] for idx in range(len(q1[i])) if idx not in indices_to_exclude]
        q3_f = [q3[i][idx] for idx in range(len(q3[i])) if idx not in indices_to_exclude]

        ax.plot(filtered_p_ins[:-1], filtered_results[:-1], marker=markers[i],
                 label="Homophily " + incentive_strategies[i].replace("_", " ").strip("'[]'"))

        print("filtered_p_ins", filtered_p_ins)
        print("q1_f", q1_f)
        print("q3_f", q3_f)

        ax.fill_between(filtered_p_ins[:-1], q1_f[:-1], q3_f[:-1], alpha=0.2)

    for i, network_type in enumerate(results_means):
        for x, strategy in enumerate(network_type):
            ax.plot(0.05, strategy, marker='o',
                     label=network_types[i].replace("_", " ").strip("'[]'") + " " + incentive_strategies[x].replace("_", " ").strip("'[]'"))

    plt.xlabel('Homophily strenght (p_in values)')
    plt.ylabel('Minimum incentive amount required')
    plt.title('Minimum incentive amount required for 95% norm abandonment for Different Incentive Strategies')
    plt.legend()
    plt.tick_params(direction="in")
    plt.savefig("Plot_Homophily_p_in_new.png")
    plt.close()

    end_t = time.time()
    elapsed_time = end_t - start_t
    print(f"Execution time: {elapsed_time} seconds")
    print("Experiment ended")