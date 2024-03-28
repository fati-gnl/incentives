"""
Exp_error_dynamics_heatmap.py

Effect of beta on norm abandonment dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.model import GameModel
from src.network_creation import create_connected_network
import multiprocessing
import src.agent_sorting as agent_sorting

import matplotlib
matplotlib.use('Agg')

def run_model_for_beta(amount_extra, beta, G, Vl, p, total_to_distribute, random_seed, incentive_strategy, sort_by_incentive_dist, size_network, model_steps):
    game_model = GameModel(
        num_agents=size_network, network=G, Vl=Vl, p=p,
        total_to_distribute=total_to_distribute, seed=random_seed, incentive_strategy=incentive_strategy,
        beta=beta, sort_by_incentive_dist=sort_by_incentive_dist, amount_extra=amount_extra
    )
    game_model.step(max_steps=model_steps)

    return game_model.pct_norm_abandonmnet[-1]

if __name__ == "__main__":
    # Initialize parameters
    size_network = 1000
    connectivity_prob = 0.05
    model_steps = 20
    Vh = 11
    Vl = 8
    p = 8
    num_runs = 3
    np.random.seed(123)
    random_seed = 124
    beta_values = [0.5, 1, 2.5, 5, 7.5, 10, 9999]

    network_type = "Erdos_Renyi"
    entitled_distribution = "Normal"
    incentive_strategy = "Random"

    # Create the network and sort nodes by incentive distribution
    G = create_connected_network(size_network, connectivity_prob, 123, Vh=Vh, gamma=True, type=network_type,
                                 entitled_distribution=entitled_distribution)
    sort_by_incentive_dist = agent_sorting.sort_by_incentive_dist(G, random_seed, incentive_strategy)

    # Lists to store data for plotting
    all_avg_pct_norm_abandonment = []

    num_runs = 10

    amounts_extra_l = np.linspace(0, 15, 2)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for amount_extra in amounts_extra_l:
        avg_pct_norm_abandonment_for_beta = []
        for beta in beta_values:
            pct_norm_abandonment_for_run = pool.starmap(run_model_for_beta, [
                (amount_extra, beta, G, Vl, p, 60000, random_seed + run, incentive_strategy, sort_by_incentive_dist,
                 size_network, model_steps)
                for run in range(num_runs)])
            avg_pct_norm_abandonment_for_beta.append(np.mean(pct_norm_abandonment_for_run))
        all_avg_pct_norm_abandonment.append(avg_pct_norm_abandonment_for_beta)
    pool.close()
    pool.join()

    # Transpose the data
    data = np.array(all_avg_pct_norm_abandonment).T

    x_ticks = amounts_extra_l

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis', aspect='auto', extent=[0, np.max(amounts_extra_l), 0, len(beta_values)])
    plt.colorbar(label='Percentage of Norm Abandonment')
    plt.ylabel('Beta')
    plt.xlabel('Addition to minimum incentive needed for changes')
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.title('Heatmap of Transition Probabilities')
    plt.savefig("333transition_prob_heatmap.png")
    plt.close()