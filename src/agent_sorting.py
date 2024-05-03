"""
agent_sorting.py

In this file, the sorted list of nodes based on a predefined Incentive strategy gets generated.

Methods:
    - sort_by_betweenness_centrality_parallel: Betweenness centrality
    - calculate_min_adopters_to_transition: Calculates the individual threshold of a node (minimum number of active neighbours that would make him transition)
    - calculate_spillover: Calculates the number of agents that will transition to the newer technology from the result of
    the activation of a target node and its neighbours.
    - sort_by_incentive_dist: Based on a predefined Incentive strategy, provides the sorted list of nodes.
"""
import networkx as nx
import numpy as np
import random
import multiprocessing as mp

def sort_by_betweenness_centrality_parallel(G):
    """
    Parallelised version to calculate the betweeness centraliry metric
    :param nx.Graph G: The graph that holds the nodes.
    :return: List of sorted nodes
    """
    with mp.Pool() as pool:
        betweenness_centrality = pool.map(nx.betweenness_centrality, [G] * len(G.nodes()))
    sorted_nodes = sorted(list(G.nodes()), key=lambda x: betweenness_centrality[x][x], reverse=True)
    return sorted_nodes

def calculate_min_adopters_to_transition(G):
    """
    This function calculates the number of neighbours of a node that should be activated for it to transition aswell without any incentives.
    :param nx.Graph G: The graph that holds the nodes.
    :return: Individual swithing threshold value
    """
    threshold = np.zeros(G.number_of_nodes())

    adjacency_matrix = nx.to_numpy_array(G)
    gamma_values = np.array(list(nx.get_node_attributes(G, 'gamma').values()))

    p = 8
    Vl = 8

    for agent_id in range(G.number_of_nodes()):
        # Get the number of neighbours of a particular node
        N = np.sum(adjacency_matrix[agent_id] > 0)

        min_adopters = float('inf')

        for i in range(N+1):
            # i = number of adopters of the new technology
            # N - i = number of neighbours who are sticking with the traditional technology
            payoff_new = gamma_values[agent_id] * N - p * (N-i)
            payoff_traditional = Vl * N - p * (i)

            if payoff_new > payoff_traditional:
                min_adopters = i
                break

        threshold[agent_id] = min_adopters

    return threshold

def calculate_spillover(G):
    """
    This function calculates the number of agents that will transition to the newer technology from the result of
    the activation of a target node and its neighbours.
    :param nx.Graph G: The graph that holds the nodes.
    :return: List of the number of spillovers resulted from the activation of each node and its neighbours
    """
    threshold = calculate_min_adopters_to_transition(G)
    adjacency_matrix = nx.to_numpy_array(G)
    num_agents = G.number_of_nodes()
    n_spillovers = np.zeros(G.number_of_nodes())

    activated_connections = []
    avg_shortest_path_lengths = []

    for node0 in range(num_agents):
        # See if any would activate
        has_activated = np.zeros(num_agents)
        agent_id = node0
        neighbors = np.nonzero(adjacency_matrix[agent_id])

        # and set their strategy to 1.
        has_activated[agent_id] = 1
        for node2 in neighbors:
            has_activated[node2] = 1

        # loop through the rest and see if they would update
        total_switched = 0
        for node in range(num_agents):
            if node != agent_id and node not in neighbors[0]:
                ind_threshold = threshold[node]
                neighbors = np.nonzero(adjacency_matrix[node])
                # number of activated neighbours
                act_neighbours = np.sum(has_activated[neighbors] == 1)
                non_activated = np.sum(has_activated[neighbors] == 0)

                assert len(neighbors[
                               0]) == act_neighbours + non_activated, "Test case failed: neighbors != act_neighbours + non_activated"

                if act_neighbours >= ind_threshold:
                    has_activated[node] = 1
                    total_switched += 1

        n_spillovers[node0] = total_switched

        activated_adjacency_matrix = np.zeros((num_agents, num_agents))
        for activated_node in np.nonzero(has_activated)[0]:
            activated_adjacency_matrix[activated_node] = adjacency_matrix[activated_node] * has_activated
        activated_connections.append(activated_adjacency_matrix)

        G1 = nx.from_numpy_array(activated_adjacency_matrix, create_using=nx.DiGraph)
        path_lengths = nx.single_source_shortest_path_length(G1, node0)
        avg_length = np.mean(list(path_lengths.values())) if path_lengths else 0

        avg_shortest_path_lengths.append(avg_length)

    return n_spillovers, avg_shortest_path_lengths

def sort_by_incentive_dist(G, seed, incentive_strategy):
    """
    This function returns a sorted list of the nodes of the network based on the incentive strategy.
    :param nx.Graph G: The graph that holds the nodes.
    :param int seed:  Seed for no_gamma number generation.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    """
    random.seed(seed)

    if incentive_strategy == "Random":
        sorted_nodes = random.sample(G.nodes(), len(G.nodes()))
    elif incentive_strategy == "Highest_degree":
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    elif incentive_strategy == "Lowest_degree":
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
    elif incentive_strategy == "Highest_gamma":
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['gamma'], reverse=True)
    elif incentive_strategy == "Lowest_gamma":
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['gamma'])
    elif incentive_strategy == "Closeness_centrality":
        sorted_nodes = sorted(G.nodes(), key=lambda x: nx.closeness_centrality(G)[x], reverse=True)
    elif incentive_strategy == "Betweenness_centrality":
        sorted_nodes = sort_by_betweenness_centrality_parallel(G)
    elif incentive_strategy == "Eigenvector_centrality":
        sorted_nodes = sorted(G.nodes(), key=lambda x: nx.eigenvector_centrality(G)[x], reverse=True)
    elif incentive_strategy == "High_local_clustering_c":
        sorted_nodes = sorted(G.nodes(), key=lambda x: nx.clustering(G, x), reverse=True)
    elif incentive_strategy =="Complex_centrality":
        _, complex_centrality = calculate_spillover(G)
        sorted_nodes = sorted(G.nodes(), key=lambda x: complex_centrality[x], reverse=True)
    else:
        raise ValueError("Invalid incentive strategy")

    return sorted_nodes