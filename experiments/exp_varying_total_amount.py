"""
Exp_varying_total_amount.py
In this file, multiple plots get generated, where the x-axis are the total money to distribute amounts.
In particular:
['time_to_95_adoption', 'number_of_spillovers', 'number_of_agents_incentivized', 'final_state', 'gini','inc_no_trans', "amount_inc_left"]
"""
import csv
import sys
import time
import src.agent_sorting as agent_sorting

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from functools import partial
from itertools import product
import warnings
import pickle

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

def run_experiment_for_strategy(incentive_strategy, total_to_distribute, G, size_network, random_seed, Vl, p,
                                model_steps, beta, sort_by_incentive_dist_l, incentive_strategies, amount_extra):
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
    index = incentive_strategies.index(incentive_strategy)
    sort_by_incentive_dist = sort_by_incentive_dist_l[index]

    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy, beta=beta, sort_by_incentive_dist=sort_by_incentive_dist,amount_extra = amount_extra
    )
    game_model.step(max_steps=model_steps)

    # Collect figures
    spillovers = game_model.spillovers
    inc_per_agent = game_model.total_incentives_ot
    gini_coeff = round(calculate_gini_coefficient(inc_per_agent),2)
    inc_no_trans = game_model.inc_but_no_transition
    incentivized= (sum(1 for amount in inc_per_agent if amount != 0))
    timesteps_95 = game_model.timesteps_95 if game_model.has_reached_95 else 0
    norm = game_model.pct_norm_abandonmnet[-1]
    incentive_amount_left = round(game_model.total_to_distribute,2)

    # Return the figures for the current strategy
    return (
        incentive_strategy, total_to_distribute, timesteps_95, spillovers, incentivized, norm,
        gini_coeff, inc_no_trans, incentive_amount_left)

if __name__ == '__main__':

    start_t = time.time()
    print("Experiment started")

    if len(sys.argv) < 3:
        print("Usage: python exp_varying_total_amount.py entitled_distribution network_type")
        sys.exit(1)

    entitled_distribution = sys.argv[1]
    network_type = sys.argv[2]

    # Constant parameters for all the experiments
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

    if network_type == "Erdos_Renyi":
        total_to_distribute_values = np.linspace(35000, 70000, 30)
    elif network_type == "Barabasi":
        total_to_distribute_values = np.linspace(35000, 80000, 30)
    elif network_type == "Homophily":
        total_to_distribute_values = np.linspace(25000, 80000, 40)

    t_l = total_to_distribute_values.tolist()

    # Different incentive strategies
    incentive_strategies = ["Highest_gamma", "Complex_centrality", "Random", "Highest_degree", "Lowest_degree", "Lowest_gamma","High_local_clustering_c", "Betweenness_centrality", "Spillover"] # []

    num_processes = mp.cpu_count()
    chunks = list(product(incentive_strategies, total_to_distribute_values))

    metrics_p = {'tm': [],'s': [],'inc': [],'count': [],'gini': [], 'inc_no_trans':[], 'left':[], 'total_to_distribute':[]}
    results_all_sim = {strategy: {metric: [[] for _ in range(len(total_to_distribute_values))] for metric in metrics_p} for strategy in incentive_strategies}

    # Run experiments for each combination of strategy and incentive level
    for i in range(num_runs):

        # Create connected network
        G = create_connected_network(size_network, connectivity_prob, random_seeds[i], Vh=Vh, gamma=True, type=network_type,
                                     entitled_distribution=entitled_distribution)

        sort_by_incentive_dist_l = []
        for str in incentive_strategies:
            sort_by_incentive_dist_l.append(agent_sorting.sort_by_incentive_dist(G, random_seeds[i], str))

        partial_func = partial(
            run_experiment_for_strategy,
            G=G,
            size_network=size_network,
            random_seed=random_seeds[i],
            Vl=Vl,
            p=p,
            model_steps=model_steps,
            beta=beta,
            sort_by_incentive_dist_l = sort_by_incentive_dist_l,
            incentive_strategies = incentive_strategies,
            amount_extra = amount_extra
        )
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(partial_func, chunks)

        for result in results:
            incentive_strategy, total_to_distribute, timesteps_95, spillovers, incentivized, count, gini, inc_no_trans, incentive_amount_left = result

            idx = t_l.index(total_to_distribute)

            results_all_sim[incentive_strategy]['tm'][idx].append(timesteps_95)
            results_all_sim[incentive_strategy]['s'][idx].append(spillovers)
            results_all_sim[incentive_strategy]['inc'][idx].append(incentivized)
            results_all_sim[incentive_strategy]['count'][idx].append(count)
            results_all_sim[incentive_strategy]['gini'][idx].append(gini)
            results_all_sim[incentive_strategy]['inc_no_trans'][idx].append(inc_no_trans)
            results_all_sim[incentive_strategy]['left'][idx].append(incentive_amount_left)
            results_all_sim[incentive_strategy]['total_to_distribute'][idx].append(total_to_distribute)

    print("results_all_sim", results_all_sim)


    def save_dict_to_pickle(data, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    results_file_name = "results_all_sim_{}_{}.json".format(network_type, entitled_distribution)
    save_dict_to_pickle(results_all_sim, results_file_name)

    def calculate_average_metrics(results_all_sim):
        """
        Calculate the average of each sublist inside each metric in results_all_sim.
        """
        for strategy, metrics in results_all_sim.items():
            for metric, data in metrics.items():
                # Calculate the average of each sublist inside the metric
                average_metric = [round(sum(sublist) / len(sublist), 2) if sublist else 0 for sublist in data]
                # Replace the original sublists with the calculated averages
                results_all_sim[strategy][metric] = average_metric

        return results_all_sim

    results_all_sim = calculate_average_metrics(results_all_sim)
    print("average results_all_sim:", results_all_sim)

    file_name = "res_f_metrics_{}_{}.txt".format(network_type, entitled_distribution)


    def save_results_to_csv(data, file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            m1 = 'Total cost used vs total number of people who have transitioned'
            m2 = 'Ratio of spillovers achieved to number of agents incentivised'
            m3 = 'Gini Coefficient'
            m4 = 'Time taken achieve a 95% transition'
            m5 = 'Percentage of total amount of money distributed'

            header = ["Strategy", m1, m2, m3, m4, m5]
            writer.writerow(header)

            for strategy_name, metrics in data.items():
                print("metrics", metrics)
                print("strategy_name", strategy_name)
                incentive_amount_left = metrics['left']
                p_norm_aband = metrics['count']
                n_agents_rv_incentive = metrics['inc']
                spillovers = metrics['s']
                gini_values = metrics['gini']
                timesteps_95 = metrics['tm']
                inc_but_no_trans = metrics['inc_no_trans']
                total_to_distribute = metrics['total_to_distribute']

                idx = p_norm_aband.index(1) if 1 in p_norm_aband else 4

                sm1 = round((total_to_distribute[idx] - incentive_amount_left[idx]) / (n_agents_rv_incentive[idx] - inc_but_no_trans[idx] + spillovers[idx]),2)
                sm2 = round(spillovers[idx] / n_agents_rv_incentive[idx],2)
                sm3 = round(gini_values[idx],2)
                sm4 = round(timesteps_95[idx],2)
                sm5 = round((total_to_distribute[idx] - incentive_amount_left[idx]) / total_to_distribute[idx],2)

                writer.writerow([strategy_name, sm1, sm2, sm3, sm4, sm5])

    save_results_to_csv(results_all_sim, file_name)

    def organize_data_for_metric(results_all_sim, metric):
        metric_data = {}
        for strategy, data in results_all_sim.items():
            metric_data[strategy] = data.get(metric, [])
        return metric_data

    def plot_and_save_results(results, filename, xdata, ylabel, incentive_strategies):
        """
        Plot and save the results of each experiment for a specific metric.
        :param dict results: Organized data for each strategy.
        :param str filename: Filename to save the plot.
        :param list xdata: Data for the x-axis.
        :param str ylabel: Label for the y-axis.
        :param list strategies: List of strategies.
        """
        markers = ['s', '^', 'D', 'v', 'o', '*', 'x', '+', '.']
        plt.figure(figsize=(8, 6))
        plt.title(ylabel)
        for i, (strategy, data) in enumerate(results.items()):
            marker = markers[i % len(markers)]
            plt.plot(xdata, data, marker=marker, label=incentive_strategies[i].replace("_", " ").strip("'[]'"))
        plt.xlabel('Incentive level')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tick_params(direction="in")
        plt.savefig(filename)
        plt.close()

    ylabels =['Time taken to achieve 95% adoption', 'Number of spillovers', 'Number of agents incentivized', 'Final state, x(F)', 'Gini Coefficient', 'Number of agents who received an incentive but did not transition', "Amount of total money left (not distributed)"]
    file_titles = ['time_to_95_adoption', 'number_of_spillovers', 'number_of_agents_incentivized', 'final_state', 'gini','inc_no_trans', "amount_inc_left"]

    for i, metric in enumerate(list(results_all_sim[incentive_strategies[0]].keys())[:-1]):
        print("metric", metric)
        metric_data = organize_data_for_metric(results_all_sim, metric)
        print("metric data",metric_data )
        plot_and_save_results(metric_data,
                              'res_f_{}_{}_{}.png'.format(network_type, entitled_distribution, file_titles[i]),
                              total_to_distribute_values, ylabels[i], list(metric_data.keys()))

    def plot_and_save_results_new(results, filename, xdata, ylabel, incentive_strategies):
        """
        Plot and save the results of each experiment for a specific metric.
        :param dict results: Organized data for each strategy.
        :param str filename: Filename to save the plot.
        :param list xdata: Data for the x-axis.
        :param str ylabel: Label for the y-axis.
        :param list strategies: List of strategies.
        """
        markers = ['s', '^', 'D', 'v', 'o', '*', 'x', '+', '.']
        plt.figure(figsize=(8, 6))
        plt.title(ylabel)
        for i, (strategy, data) in enumerate(results.items()):
            marker = markers[i % len(markers)]
            print("xdata", xdata)
            print("data", data)
            print("data new", [a / b for a, b in zip(data, xdata)])
            plt.plot(xdata, [a / b for a, b in zip(data, xdata)], marker=marker, label=incentive_strategies[i].replace("_", " ").strip("'[]'"))
        plt.xlabel('Incentive level')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tick_params(direction="in")
        plt.savefig(filename)
        plt.close()

    ylabel_new = ["Amount of total money not distributed (%)"]
    file_title_new = ["percentage_inc_left"]
    for i, metric in enumerate(list(results_all_sim[incentive_strategies[0]].keys())[:-1]):
        metric_data = organize_data_for_metric(results_all_sim, metric)
        print("metric_Data", metric_data)
        if i == 6:
            plot_and_save_results_new(metric_data,
                                'res_f_{}_{}_{}.png'.format(network_type, entitled_distribution, file_title_new),
                                total_to_distribute_values, ylabel_new, list(metric_data.keys()))

    et = time.time()
    elapsed_time = et - start_t
    print(f"Execution time: {elapsed_time} seconds")
    print("Experiment ended")