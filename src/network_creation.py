"""
network_creation.py

This file contains functions for creating networks and setting initial node strategies based on various initialization methods.

Functions:
    - add_homophily_edges: Add edges to a graph based on homophily condition; how close the gamma values are.
    - set_strategies: Initialize the strategies for each agent based on the provided parameters.
    - create_connected_network: Create a connected network with optional homophily and gamma distribution.
"""

import networkx as nx
import random
import warnings
from src.model import GameModel

def add_homophily_edges(G, gamma_values, homophily_strength):
    """
    Add edges to a graph based on homophily condition; how close to each other the gamma values are.
    :param nx.Graph G: The graph to add edges to.
    :param dict gamma_values: A dictionary mapping node indices to gamma values.
    :param float homophily_strength: Strength of homophily effect.
    :return:
    """
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2 and not G.has_edge(node1, node2):
                gamma_diff = abs(gamma_values[node1] - gamma_values[node2])
                probability = 1 - homophily_strength * gamma_diff
                if random.uniform(0, 1) < probability:
                    G.add_edge(node1, node2)

def set_strategies(G, count, node_degree, seed, gamma, initialisation):
    """
    Initialise the strategies for each agent based on the provided parameters
    :param nx.Graph G: The graph that holds the nodes.
    :param count: Number of nodes to set their initial strategies to "Adopt New Technology"
    :param node_degree: Degree of the nodes whose strategies are being affected.
    :param Boolean gamma: whether or not the Vh values will be normally distributed accross the nodes. Otherwise, gamma = Vh.
    :param String initialisation: How the nodes will be initialised: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma", "By_specified_degree"
    :return: The degree of the nodes
    """
    node_degrees = dict(G.degree())
    distinct_degrees = set(node_degrees.values())
    random.seed(seed)

    if (gamma == False and initialisation == "Highest_gamma") or (gamma == False and initialisation == "Lowest_gamma"):
        warnings.warn("You are trying to initialize your nodes with a strategy that depends on the gamma values. "
                      "However, your parameter gamma is set to False, so all nodes will have the same gamma value, "
                      "meaning this strategy will be the same as random initialization.")

    if initialisation == "Random":
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
        for _ in range(count):
            node = random.choice([node for node in G.nodes if G.nodes[node]['strategy'] != "Adopt New Technology"])
            strategy = "Adopt New Technology"
            G.nodes[node]['strategy'] = strategy

    elif initialisation == "Highest_degree":
        sorted_nodes_by_degree = sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
        for node in sorted_nodes_by_degree[:count]:
            strategy = "Adopt New Technology"
            G.nodes[node]['strategy'] = strategy

    elif initialisation == "Lowest_degree":
        sorted_nodes_by_degree = sorted(G.nodes, key=lambda x: G.degree(x))
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
        for node in sorted_nodes_by_degree[:count]:
            strategy = "Adopt New Technology"
            G.nodes[node]['strategy'] = strategy

    elif initialisation == "Highest_gamma":
        sorted_nodes_by_gamma= sorted(G.nodes, key=lambda x: G.nodes[x]['gamma'], reverse=True)
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
        for node in sorted_nodes_by_gamma[:count]:
            strategy = "Adopt New Technology"
            G.nodes[node]['strategy'] = strategy

    elif initialisation == "Lowest_gamma":
        sorted_nodes_by_gamma = sorted(G.nodes, key=lambda x: G.nodes[x]['gamma'])
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')
        for node in sorted_nodes_by_gamma[:count]:
            strategy = "Adopt New Technology"
            G.nodes[node]['strategy'] = strategy

    elif initialisation == "By_specified_degree":
        nx.set_node_attributes(G, "Stick to Traditional", 'strategy')

        # Look for the nodes that belong to the specified "node_degree" parameter
        nodes_with_degree = [node for node, degree in node_degrees.items() if degree == node_degree]
        if len(nodes_with_degree) < count:
            count = len(nodes_with_degree)

        # For those nodes, set the strategy to "Adopt New Technology" if they have been randomly selected out of the total "count" speficied.
        nodes_to_change = random.sample(nodes_with_degree, count)
        for node in nodes_to_change:
            degree_of_node = G.degree(node)
            print(f"Degree of the node {node} changing: {degree_of_node}")
            G.nodes[node]['strategy'] = "Adopt New Technology"

    # Print distinct degrees and their counts
    print("Distinct degrees of nodes and their counts:")
    for degree in distinct_degrees:
        count = list(node_degrees.values()).count(degree)
        print(f"Degree {degree}: {count} nodes")

    return node_degrees

def create_connected_network(size, connectivity, seed, Vh, homophily=False, homophily_strength=0.25, count=0, node_degree=0, gamma=False, initialisation="Highest_degree"):
    """"
    Create a connected no_gamma network.
    :param int size: Number of nodes in the network.
    :param float connectivity: Probability of edge creation between nodes.
    :param int seed:  Seed for no_gamma number generation.
    :param float Vh: High reward for selecting their preferred strategy.
    :param Boolean homophily: Whether or not the network will be structured based on the similarity of the gamma values of ndoes
    :param float homophily_strength
    :param int count: number of nodes to initialise with the "Adopt New Technology" strategy
    :param node_degree: specify the degrees that you want to initialise
    :param Boolean gamma: whether or not the Vh values will be normally distributed accross the nodes. Otherwise, gamma = Vh.
    :param String initialisation: How the nodes will be initialised: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    :return: Connected no_gamma network of type networkx.Graph.
    """
    if homophily:
        G = nx.Graph()
        for node in range(size):
            G.add_node(node)
        random.seed(seed)
        if gamma:
            gamma_values = {
                node: random.choice(GameModel.generate_truncated_normal(mean=Vh, lower_bound=Vh - 2, upper_bound=Vh + 2, size=1000))
                for node in G.nodes}
        else:
            gamma_values = {node: Vh for node in G.nodes}
        nx.set_node_attributes(G, gamma_values, 'gamma')
        add_homophily_edges(G, gamma_values, homophily_strength)
    else:
        G = nx.erdos_renyi_graph(size, connectivity, seed=seed)
        random.seed(seed)
        # If there are nodes without edges, connect them to the network.
        while not nx.is_connected(G):
            nodes_to_connect = list(nx.isolates(G))
            if not nodes_to_connect:
                break
            node1 = nodes_to_connect[0]
            node2 = random.choice(list(G.nodes - {node1}))
            G.add_edge(node1, node2)

        if gamma:
            gamma_values = {
                node: random.choice(GameModel.generate_truncated_normal(mean=Vh, lower_bound=Vh-2, upper_bound=Vh+2, size=1000))
                for node in G.nodes}
        else:
            gamma_values = {node: Vh for node in G.nodes}
        nx.set_node_attributes(G, gamma_values, 'gamma')

    # Initialise the agents strategies
    node_degrees = set_strategies(G, count, node_degree, seed, gamma, initialisation)
    return G, node_degrees