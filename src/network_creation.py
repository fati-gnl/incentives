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

#matplotlib.use('Agg')

def create_homophily_network(gamma_values, seed):
    """
    Creates a homophily network based on the selected gamma distribution.
    :param dict gamma_values: A dictionary mapping node indices to gamma values.
    :param int seed: Seed for random number generator.
    """
    sorted_gamma_v = np.sort(gamma_values)
    n_chunks = 20

    # Divide the sorted gamma list into n chunks
    chunks = np.array_split(sorted_gamma_v, n_chunks)
    chunk_sizes = [len(chunk) for chunk in chunks]

    p_itself = 0.8
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

    return G

def network_type(size, connectivity, seed, type):
    if type == "Erdos_Renyi":
        G = nx.erdos_renyi_graph(size, connectivity, seed=seed)
    elif type == "Barabasi":
        G = nx.barabasi_albert_graph(size, int(0.025*size))
        #G = nx.barabasi_albert_graph(size, 3)
    elif type == "Homophily":
        G = nx.Graph()
        for node in range(size):
            G.add_node(node)
    else:
        raise ValueError("Please specify a valid network type: Erdos_Renyi, Barabasi or Homophily.")
    return G

def create_connected_network(size, connectivity, seed, Vh, gamma, type, entitled_distribution):
    """"
    Create a connected no_gamma network.
    :param int size: Number of nodes in the network.
    :param float connectivity: Probability of edge creation between nodes.
    :param int seed:  Seed for no_gamma number generation.
    :param float Vh: High reward for selecting their preferred strategy.
    :param Boolean gamma: whether or not the Vh values will be normally distributed accross the nodes. Otherwise, gamma = Vh.
    :param String type: Erdos_Renyi, Barabasi, Homophily
    :param String entitled_distribution: Type of distribution for the gamma distributions.
    :return: Connected no_gamma network of type networkx.Graph.
    """

    G = network_type(size, connectivity, seed, type)

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

    if gamma:
        gamma_values = np.random.choice(
            GameModel.generate_distribution(lower_bound=Vh-2, upper_bound=Vh+2, size=10000, entitled_distribution=entitled_distribution), size=size)
    else:
        gamma_values = np.full(size, Vh)

    nx.set_node_attributes(G, dict(zip(G.nodes, gamma_values)), 'gamma')
    nx.set_node_attributes(G, "Stick to Traditional", 'strategy')

    if type == "Homophily":
        #add_homophily_edges(G, gamma_values, entitled_distribution)
        G = create_homophily_network(gamma_values, seed)

    node_degrees = dict(G.degree())
    distinct_degrees = set(node_degrees.values())

    count_list=[]

    # Print distinct degrees and their counts
    print("Distinct degrees of nodes and their counts:")
    for degree in distinct_degrees:
        count = list(node_degrees.values()).count(degree)
        count_list.append(count)
        print(f"Degree {degree}: {count} nodes")

    average_degree = np.mean(list(dict(G.degree()).values()))
    print("Average network degree:", average_degree)

    total_links = G.number_of_edges()
    print("Total number of links:", total_links)

    return G

#create_connected_network(size=1000, connectivity=0.05, seed=15, Vh=11, gamma=True, type="Homophily", entitled_distribution="Normal")
