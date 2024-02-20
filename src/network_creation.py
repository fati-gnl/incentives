"""
network_creation.py

This file contains functions for creating networks and setting initial node strategies based on various initialization methods.

Functions:
    - add_homophily_edges: Add edges to a graph based on homophily condition; how close the gamma values are.
    - set_strategies: Initialize the strategies for each agent based on the provided parameters.
    - create_connected_network: Create a connected network with optional homophily and gamma distribution.
    - incentive_distribution: Distributes the incentive amount to a specified number of individuals based on the chosen incentive distribution strategy.
"""
import networkx as nx
import random
import numpy as np
from src.model import GameModel

def add_homophily_edges(G, gamma_values):
    """
    Add edges to a graph based on homophily condition; how close to each other the gamma values are.
    :param nx.Graph G: The graph to add edges to.
    :param dict gamma_values: A dictionary mapping node indices to gamma values.
    :param float homophily_strength: Strength of homophily effect.
    :return:
    """
    homophily_strength = 8.2
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2 and not G.has_edge(node1, node2):
                gamma_diff = abs(gamma_values[node1] - gamma_values[node2])
                probability = 1 - homophily_strength * gamma_diff
                if random.uniform(0, 1) < probability:
                    G.add_edge(node1, node2)

def network_type(size, connectivity, seed, type):
    if type == "Erdos_Renyi":
        G = nx.erdos_renyi_graph(size, connectivity, seed=seed)
    elif type == "Barabasi":
        G = nx.barabasi_albert_graph(size, int(0.026*size))
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
    :return: Connected no_gamma network of type networkx.Graph.
    """

    G = network_type(size, connectivity, seed, type)

    random.seed(seed)
    np.random.seed(seed)

    # If there are nodes without edges, connect them to the network.
    if type != "Homophily":
        print("has entered")
        while not nx.is_connected(G):
            nodes_to_connect = list(nx.isolates(G))
            if not nodes_to_connect:
                break
            node1 = nodes_to_connect[0]
            node2 = random.choice(list(G.nodes - {node1}))
            G.add_edge(node1, node2)

    # GameModel.generate_truncated_normal(mean=[Vh][12, 10], lower_bound=Vh-2, upper_bound=Vh+2, size=10000, sd=[1][0.4, 0.6]),
    # size=size)

    if gamma:
        gamma_values = np.random.choice(
            GameModel.generate_truncated_normal(lower_bound=Vh-2, upper_bound=Vh+2, size=10000, entitled_distribution=entitled_distribution), size=size)
    else:
        gamma_values = np.full(size, Vh)

    nx.set_node_attributes(G, dict(zip(G.nodes, gamma_values)), 'gamma')
    nx.set_node_attributes(G, "Stick to Traditional", 'strategy')

    if type == "Homophily":
        add_homophily_edges(G, gamma_values)

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


# create_connected_network(size=1000, connectivity=0.05, seed=123, Vh=11, gamma=True, type="Homophily", entitled_distribution="Uniform")
