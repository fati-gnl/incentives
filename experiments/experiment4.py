"""
Experiment4.py

In this file, we will investigate the minimum incentive required to cause a full transition in the population.
"""
import matplotlib.pyplot as plt
from src.network_creation import create_connected_network
from src.model import GameModel
import numpy as np

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
p = 10

initiators = 0.2

incentive_strategy = "Random"
initialisation = "Lowest_degree"
incentive_count_values = np.linspace(0.1, 0.45, 25)

results = []

last_incentive = 0

for incentive_count in reversed(incentive_count_values):

    incentive_count = int(incentive_count * size_network)
    print(incentive_count)
    incentive_amount = last_incentive
    break_outer = False

    G, node_degrees = create_connected_network(
        size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
        initiators=int(initiators*size_network), node_degree=0, gamma=True,
        initialisation=initialisation, incentive_count=incentive_count, incentive_amount=incentive_amount,
        incentive_strategy=incentive_strategy)

    model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, p=p)

    for step in range(model_steps):
        model.step()

    pct_norm_abandonment = model.pct_norm_abandonmnet[-1]

    if pct_norm_abandonment != 100:
        incentive_amount += 1
        print("not 100, so starting while loop")
        while True:
            G, node_degrees = create_connected_network(
                size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
                initiators=int(initiators*size_network), node_degree=0, gamma=True,
                initialisation=initialisation, incentive_count=incentive_count, incentive_amount=incentive_amount,
                incentive_strategy=incentive_strategy)

            model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, p=p)

            for step in range(model_steps):
                model.step()

            pct_norm_abandonment = model.pct_norm_abandonmnet[-1]

            # Check if everyone has abandoned the norm
            if pct_norm_abandonment == 100:
                print("has entered, successful result")
                results.append((incentive_count*incentive_amount, incentive_amount))
                break_outer = True
                last_incentive = incentive_amount
                break
            else:
                incentive_amount += 1
                print("current incentive amount = " + str(incentive_amount))

    if break_outer:
        break

    if pct_norm_abandonment == 100 and incentive_count == reversed(incentive_count_values)[0]:
        results.append((incentive_count, 0))
    else:
        print("pct is == 100 for incentive_count " + str(incentive_count))

# Plot the results
incentive_count_values_plot = [ic for ic, ia in results]
incentive_amount_values = [ia for ic, ia in results]

file_path = "results_experiment4.txt"
np.savetxt(file_path, results, header="Incentive Initiators Count vs Incentive Amount", comments="")

# Plot the results
plt.plot(incentive_count_values_plot, incentive_amount_values, marker='o')

plt.ylabel("Incentive Amount")
plt.xlabel("Percentage who receive an incentive")
plt.title("Incentive Amount as a Function of Incentive Initiators Count (for p={})".format(p))
plt.show()
