import networkx as nx

def create_connected_network(size, connectivity, seed):
    """
    Create a connected random network.
    :param int size: Number of nodes in the network.
    :param float connectivity: Probability of edge creation between nodes.
    :param int seed:  Seed for random number generation.
    :return: Connected random network of type networkx.Graph.
    """
    G = nx.erdos_renyi_graph(size, connectivity, seed=seed)

    while not nx.is_connected(G):
        nodes_to_connect = nx.non_edges(G, seed=seed)
        edge_to_add = next(nodes_to_connect)
        G.add_edge(*edge_to_add)

    return G