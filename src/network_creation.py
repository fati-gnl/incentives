"""
network_creation.py

This file contains functions for creating networks and setting initial node strategies based on various initialization methods.

Functions:
    - add_homophily_edges: Add edges to a graph based on homophily condition; how close the gamma values are.
    - create_connected_network: Create a connected network with optional homophily and gamma distribution.
    - network_type: Initiallises a network based on the network tyoe
"""
import networkx as nx
import random
import numpy as np
from src.model import GameModel
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def create_homophily_network(gamma_values, seed, p_in, average_degree):
    """
    Creates a homophily network based on the selected gamma distribution.
    :param dict gamma_values: A dictionary mapping node indices to gamma values.
    :param int seed: Seed for random number generator.
    """
    sorted_gamma_v = np.sort(gamma_values)
    N = len(sorted_gamma_v)
    n_chunks = int(N / average_degree)

    # Divide the sorted gamma list into n chunks
    chunks = np.array_split(sorted_gamma_v, n_chunks)
    chunk_sizes = [len(chunk) for chunk in chunks]

    p_itself = p_in
    p_outside_block = 1 - p_itself
    probabilities = np.zeros((n_chunks, n_chunks))
    for i in range(n_chunks):
        for j in range(n_chunks):
            if i == j:
                probabilities[i, j] = p_itself
            else:
                probabilities[i, j] = p_outside_block / (n_chunks - 1)

    G = nx.stochastic_block_model(chunk_sizes, probabilities, seed=seed)
    nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
    nx.set_node_attributes(G, dict(zip(G.nodes, sorted_gamma_v)), 'gamma')

    #node_colors = [sorted_gamma_v[node] for node in G.nodes()]
    #node_labels = {node: f"{gamma:.2f}" for node, gamma in zip(G.nodes(), gamma_values)}

    #pos = nx.spring_layout(G)  # Layout for visualization
    #plt.figure(figsize=(10, 6))
    #plt.title('Homophily Network Visualization')
    #nx.draw(G, pos, node_color=node_colors)
    #nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=8)
    #plt.savefig("homophily_n.png")
    #plt.close()

    node_index = 0
    for chunk_index, chunk in enumerate(chunks):
        for node_value in chunk:
            node = list(G.nodes())[node_index]
            G.nodes[node]['chunk_number'] = chunk_index
            node_index += 1

    return G

def network_type(size, seed, type, average_degree):
    """
    :param int size: Number of nodes in the network.
    :param int seed:  Seed for no_gamma number generation.
    :param String type: Erdos_Renyi, Barabasi, Homophily
    :param int average_degree: Average degree of a network
    :return:
    """
    if type == "Erdos_Renyi":
        G = nx.erdos_renyi_graph(size, (average_degree / (size - 1)), seed=seed)
    elif type == "Barabasi":
        G = nx.barabasi_albert_graph(size,int(average_degree / 2))
        #G = nx.barabasi_albert_graph(size, 3)
    elif type == "Homophily":
        G = nx.Graph()
        for node in range(size):
            G.add_node(node)
    else:
        raise ValueError("Please specify a valid network type: Erdos_Renyi, Barabasi or Homophily.")
    return G

def create_connected_network(size, seed, Vh, type, entitled_distribution, width_percentage, Vl, p_in = 0.8, average_degree=50):
    """"
    Create a connected no_gamma network.
    :param int size: Number of nodes in the network.
    :param int seed:  Seed for no_gamma number generation.
    :param float Vh: High reward for selecting their preferred strategy.
    :param float Vl: Low reward for not selecting their preferred strategy.
    :param float width_percentage: Distance between the distribution of gammas and Vl
    :param String type: Erdos_Renyi, Barabasi, Homophily
    :param String entitled_distribution: Type of distribution for the gamma distributions.
    :param float p_in: Probability of connecting within a chunk
    :param int average_degree: Average degree of a network
    :return: Connected no_gamma network of type networkx.Graph.
    """

    G = network_type(size, seed, type, average_degree)

    random.seed(seed)
    np.random.seed(seed)

    # If there are nodes without edges, connect them to the network.
    if type != "Homophily":
        while not nx.is_connected(G):
            nodes_to_connect = list(nx.isolates(G))
            if not nodes_to_connect:
                break
            node1 = nodes_to_connect[0]
            node2 = random.choice(list(G.nodes - {node1}))
            G.add_edge(node1, node2)

    gamma_values = np.random.choice(
        GameModel.generate_distribution(mean=Vh, min_value=Vl,width_percentage=width_percentage,entitled_distribution = entitled_distribution), size=size)

    if type != "Homophily":
        nx.set_node_attributes(G, dict(zip(G.nodes, gamma_values)), 'gamma')

        unique_gamma_values = sorted(set(gamma_values))

        n_chunks = int(size / average_degree)
        size_chunks = size / n_chunks

        chunk_number = 0
        count = 0
        for value in unique_gamma_values:
            nodes_with_value = [node for node, data in G.nodes(data=True) if data['gamma'] == value]
            for node in nodes_with_value:
                G.nodes[node]['chunk_number'] = chunk_number
                count += 1
                # Check if reached size_chunks limit
                if count >= size_chunks:
                    count = 0
                    chunk_number += 1

    nx.set_node_attributes(G, "Stick to Traditional", 'strategy')

    if type == "Homophily":
        #add_homophily_edges(G, gamma_values, entitled_distribution)
        G = create_homophily_network(gamma_values, seed, p_in, average_degree)

    node_degrees = dict(G.degree())
    distinct_degrees = set(node_degrees.values())

    count_list=[]

    # Print distinct degrees and their counts
    #print("Distinct degrees of nodes and their counts:")
    for degree in distinct_degrees:
        count = list(node_degrees.values()).count(degree)
        count_list.append(count)
        #print(f"Degree {degree}: {count} nodes")

    average_degree = np.mean(list(dict(G.degree()).values()))
    #print("Average network degree:", average_degree)

    total_links = G.number_of_edges()
    #print("Total number of links:", total_links)

    return G

G = create_connected_network(size=1000, seed=15, Vh=11, type="Erdos_Renyi", entitled_distribution="BiModal", width_percentage = 0.675,
                                 Vl=8, p_in=0.05)
print("E", nx.average_clustering(G))

G = create_connected_network(size=1000, seed=15, Vh=11, type="Barabasi", entitled_distribution="BiModal", width_percentage = 0.675,
                                 Vl=8, p_in=0.05)
print("B", nx.average_clustering(G))



"""
p_ins = np.round(np.linspace(0, 1, 10), 2)
assortativity_coefficients = []
q1 = []
q3 = []
random_seeds = [np.random.randint(10000) for _ in range(20)]

for p_in in p_ins:

    avg = []

    for seed in random_seeds:

        G = create_connected_network(size=1000, seed=seed, Vh=9, type="Homophily", entitled_distribution="BiModal", width_percentage = 1,
                                     Vl=8, p_in=p_in)

        assortativity = nx.attribute_assortativity_coefficient(G, 'chunk_number')
        avg.append(assortativity)

    assortativity_coefficients.append(np.mean(avg))
    q1.append(np.percentile(avg, 25))
    q3.append(np.percentile(avg, 75))

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(p_ins, assortativity_coefficients, marker='o', linestyle='-', color='b')
ax.fill_between(p_ins, q1, q3, alpha=0.2)

G = create_connected_network(size=1000, seed=15, Vh=11, type="Erdos_Renyi", entitled_distribution="BiModal", width_percentage = 0.675,
                                 Vl=8, p_in=p_in)
a_erdos = nx.attribute_assortativity_coefficient(G, 'chunk_number')
plt.plot(0.05, a_erdos, marker='o', linestyle='-', color='g', label="Erdos-Renyi BiModal")

G = create_connected_network(size=1000, seed=15, Vh=11, type="Barabasi", entitled_distribution="BiModal", width_percentage = 0.675,
                                 Vl=8, p_in=p_in)
a_bara = nx.attribute_assortativity_coefficient(G, 'chunk_number')
plt.plot(0.05, a_bara, marker='o', linestyle='-', color='red', label="Barabasi BiModal")

plt.title('Attribute Assortativity Coefficient vs. p_in (Homophily)')
plt.xlabel('p_in values')
plt.ylabel('Attribute Assortativity Coefficient')
plt.legend()
plt.tick_params(direction="in")
plt.savefig("Assortativity_Coefficients")
"""