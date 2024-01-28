import matplotlib.pyplot as plt
import src.network_creation
from src.model import GameModel
from tqdm import tqdm
import numpy as np

# Parameters
size_network = 1000
connectivity_prob = 0.05
random_seed = 123
model_steps = 15
Vh = 11
Vl = 8
#p_values = [4,4.5,5,5.5,6,6.5,7,7.5,8,8.8,9,9.5,10,10.5,11]
p_values = [5.5,6,6.5,7,7.5,8,8.8,9,9.5,10,10.5,11,11.5,12,12.5]

results = []

def calculate_tipping_threshold(Vl, Vh, p):
    return (-Vh + Vl + p)/ (2*p)

for p in p_values:
    tipping_threshold = calculate_tipping_threshold(Vl, Vh, p)
    print("tipping threshold" + str(tipping_threshold))
    break_outer = False
    initiator_count = int((tipping_threshold - 0.025) * size_network)
    while True:
        # Check that for the first value that we are checking, the system is not already fully transitioned
        print("initiator count" + str(initiator_count))

        G, node_degrees = network_creation.create_connected_network(
            size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
            count=initiator_count, node_degree=0, gamma=False, initialisation="Highest_node")


        model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, Vh=Vh, p=p)

        for step in range(model_steps):
            model.step()

        pct_norm_abandonment = model.pct_norm_abandonmnet[-1]

        if(pct_norm_abandonment != 100):
            print("not 100, so starting 2 while loop")
            while True:
                G, node_degrees = network_creation.create_connected_network(
                        size_network, connectivity_prob, random_seed, Vh=Vh, homophily=False, homophily_strength=0.01,
                        count=initiator_count, node_degree=0
                    )

                model = GameModel(num_agents=size_network, network=G, node_degrees=node_degrees, Vl=Vl, Vh=Vh, p=p)

                for step in range(model_steps):
                    model.step()

                pct_norm_abandonment = model.pct_norm_abandonmnet[-1]

                # Check if everyone has abandoned the norm
                if pct_norm_abandonment == 100:
                    print("has entered, successful result")
                    results.append(((initiator_count / size_network), tipping_threshold))
                    break_outer = True
                    break
                else:
                    initiator_count += 1
                    print("current initiator count = " + str(initiator_count))

        if break_outer:
            break

        if initiator_count > 0:
            initiator_count -= int((0.005 * size_network))
        else:
            results.append((0, tipping_threshold))
            break

# Plot the results
initiator_probabilities = [initiator_count for initiator_count, tipping_threshold in results]
tipping_thresholds = [tipping_threshold for initiator_count, tipping_threshold in results]

file_path = "results.txt"
np.savetxt(file_path, results, header="Initiator Probability Tipping Threshold", comments="")

# Plot the results
#plt.plot(initiator_probabilities, tipping_thresholds, marker='o')
plt.plot(tipping_thresholds, initiator_probabilities, marker='o')

plt.ylabel("Initiator Probability")
plt.xlabel("Tipping Threshold")
plt.title("Tipping Threshold as a Function of Initiator Probability")
plt.show()