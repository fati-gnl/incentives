"""
network_degree_count.py

With the use of this file, you can get a list for a certain network size and connectivity probability
that details how many nodes have a particular degree.
"""

import random
import networkx as nx
import os

def create_connected_network(size, connectivity, seed1):
    """
    Returns a list for a certain network size and connectivity probability
    that details how many nodes have a particular degree.
    :param int size: Number of nodes in the network.
    :param float connectivity: Probability of edge creation between nodes.
    :param int seed1:  Seed for number generation.
    :return: degree, count_list
    """
    random.seed(seed1)
    G = nx.erdos_renyi_graph(size, connectivity, seed=seed1)

    # If there are nodes without edges, connect them to the network.
    while not nx.is_connected(G):
        nodes_to_connect = list(nx.isolates(G))
        if not nodes_to_connect:
            break
        node1 = nodes_to_connect[0]
        node2 = random.choice(list(G.nodes - {node1}))
        G.add_edge(node1, node2)

    node_degrees = dict(G.degree())
    distinct_degrees = set(node_degrees.values())

    count_list=[]

    # Print distinct degrees and their counts
    print("Distinct degrees of nodes and their counts:")
    for degree in distinct_degrees:
        count = list(node_degrees.values()).count(degree)
        count_list.append(count)
        print(f"Degree {degree}: {count} nodes")

    # Save output to a text file
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    file_path = os.path.join(results_folder, 'output.txt')

    with open(file_path, 'w') as file:
        for degree, count in zip(distinct_degrees, count_list):
            file.write(f"Degree {degree}: {count} nodes\n")

    print(f"Results saved to: {file_path}")

    return degree, count_list


create_connected_network(size=1000, connectivity=0.05, seed1=123)