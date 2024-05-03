"""
test_avg_degree_network.py
This file verifies that all of the possible distributions have a similar average degree.
"""
import numpy as np
import unittest
from src.network_creation import create_connected_network

class Tests(unittest.TestCase):

    def test_network_degree(self):
        """
        Test to verify that all of the possible network combinations have a similar average degree
        """
        types = ["Barabasi", "Erdos_Renyi", "Homophily"]
        distributions = ["BiModal", "Normal", "Uniform"]

        avg_degree = 20
        threshold = avg_degree

        for network_type in types:
            for distribution in distributions:
                G = create_connected_network(size=1000, connectivity=0.05, seed=15, Vh=11, gamma=True, type=network_type, entitled_distribution=distribution, p_in = 0.8, average_degree = avg_degree)
                average_degree = np.mean(list(dict(G.degree()).values()))
                print("average_degree", average_degree)
                self.assertTrue((average_degree - 2) <= threshold) and ((average_degree + 2) >= threshold), f"Average network degree deviates by more than {threshold} units for {network_type} network with {distribution} distribution."
        print("All tests passed!")

if __name__ == '__main__':
    unittest.main()