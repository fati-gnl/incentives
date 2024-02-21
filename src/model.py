"""
model.py

This file defines the GameModel class, a model representing the dynamics of agents in a networked game scenario.
Agents interact with their neighbors, updating their strategies based on payoff calculations and individual thresholds.

Methods:
    - generate_distribution: Generate Vh values from a given distribution.
    - step: Execute different step of the model until a maximum predefined step has reached, or no changes have been found for a consecutive number of rounds.
"""
import networkx as nx
import numpy as np
from scipy.stats import truncnorm
import random

def sort_by_incentive_dist(G, seed, incentive_strategy):
    """
    This function returns a sorted list of the nodes of the network based on the incentive strategy.
    :param nx.Graph G: The graph that holds the nodes.
    :param int seed:  Seed for no_gamma number generation.
    :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
    """
    random.seed(seed)

    # Filter out nodes with strategy "Adopt New Technology"
    filtered_nodes = [node for node, data in G.nodes(data=True) if data['strategy'] == "Stick to Traditional"]

    if incentive_strategy == "Random":
        sorted_nodes = random.sample(filtered_nodes, len(filtered_nodes))
    elif incentive_strategy == "Highest_degree":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: G.degree(x), reverse=True)
    elif incentive_strategy == "Lowest_degree":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: G.degree(x))
    elif incentive_strategy == "Highest_gamma":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: G.nodes[x]['gamma'], reverse=True)
    elif incentive_strategy == "Lowest_gamma":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: G.nodes[x]['gamma'])
    elif incentive_strategy == "Closeness_centrality":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: nx.closeness_centrality(G)[x], reverse=True)
    elif incentive_strategy == "Betweenness_centrality":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: nx.betweenness_centrality(G)[x], reverse=True)
    elif incentive_strategy == "Eigenvector_centrality":
        sorted_nodes = sorted(filtered_nodes, key=lambda x: nx.eigenvector_centrality(G)[x], reverse=True)
    else:
        raise ValueError("Invalid incentive strategy")

    return sorted_nodes

class GameModel():
    def __init__(self, num_agents, network, Vl, p, total_to_distribute, seed, incentive_strategy, beta):
        """
        Initialize a GameModel.
        :param int num_agents: Number of agents in the model.
        :param network: Parameters for generating the network (e.g., size, connectivity).
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        :param int total_to_distribute: Amount of incentive to distribute
        :param int seed: Seed for random number generations
        :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
        :param int beta: rate of transition between the strategies (sigmoid function)
        """
        self.num_agents = num_agents
        self.network = network
        self.pos = nx.spring_layout(self.network)
        self.seed = seed
        self.total_to_distribute = total_to_distribute
        self.incentive_strategy = incentive_strategy
        self.p = p
        self.Vl = Vl
        self.beta = beta

        np.random.seed(self.seed)

        # Initialize arrays with the agents properties
        self.gamma_values = np.array(list(nx.get_node_attributes(self.network, 'gamma').values()))
        self.current_strategies = np.array(list(nx.get_node_attributes(self.network, 'strategy').values()))
        self.incentives = np.zeros(num_agents)
        self.total_incentives_ot = np.zeros(num_agents)
        self.probabilities_informed = np.zeros(num_agents)

        # Get the adjacent matrix for the neighbours
        self.adjacency_matrix = nx.to_numpy_matrix(self.network)

        self.sorted_nodes = sort_by_incentive_dist(self.network, self.seed, self.incentive_strategy)

        num_nodes = len(self.sorted_nodes)
        for i, node in enumerate(self.sorted_nodes):
            self.probabilities_informed[node] = (num_nodes - i) / num_nodes

        # Metrics that I am keeping track of for the experiments
        self.pct_norm_abandonmnet = []
        self.number_of_incentivised = 0
        self.incentive_amounts = []
        self.transition_probs = []
        self.sigmoid_inputs = []
        self.spillovers = 0
        self.timesteps_95 = 0

    @staticmethod
    def generate_distribution(lower_bound: float, upper_bound: float, size: int, entitled_distribution: str) -> np.ndarray:
        """
        Generate values from a truncated normal distribution.
        :param float lower_bound: Lower bound for the truncated distribution.
        :param float upper_bound: Upper bound for the truncated distribution.
        :param int size: Number of values to generate.
        :param str entitled_distribution: Type of distribution from ["Uniform", "Normal", "BiModal"]
        :return: Generated values from the truncated normal distribution.
        """
        if entitled_distribution == "Uniform":
            values = np.random.uniform(low=9, high=13, size=100000)

        elif (entitled_distribution == "Normal"):
            sd = [1]
            mean = [11]
            a1, b1 = (lower_bound - mean[0]) / sd[0], (upper_bound - mean[0]) / sd[0]
            values = truncnorm.rvs(a1, b1, loc=mean[0], scale=sd[0], size=size)

        elif (entitled_distribution == "BiModal"):
            sd = [0.4, 0.6]
            mean = [12,10]
            a1, b1 = (lower_bound - mean[0]) / sd[0], (upper_bound - mean[0]) / sd[0]
            values = truncnorm.rvs(a1, b1, loc=mean[0], scale=sd[0], size=size)

            a2, b2 = (lower_bound - mean[1]) / sd[1], (upper_bound - mean[1]) / sd[1]
            values2 = truncnorm.rvs(a2, b2, loc=mean[1], scale=sd[1], size=size)

            values = np.concatenate((values, values2))
        else:
            raise ValueError("Invalid entitled distribution strategy")

        return values

    def payoff_current_choice(self, agent_id: int, num_stick_to_traditional: int, num_adopt_new_tech:int, N:int) -> float:
        """
        Calculate the payoff for playing the current agents strategy.
        :param int agent_id: Index of the current agent
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        :param int N: Number of neighbors
        :return: Payoff for the current strategy
        """
        if self.current_strategies[agent_id] == "Adopt New Technology":
            payoff = self.gamma_values[agent_id] * N - self.p * num_stick_to_traditional
        else:
            payoff = self.Vl * N - self.p * num_adopt_new_tech
        return payoff

    def update_strategy(self, agent_id: int, num_stick_to_traditional: int, num_adopt_new_tech: int):
        """
        Update the agent's strategy based on the calculated payoff and individual threshold.
        :param int agent_id: Index of the current agent
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        """
        N: int = np.sum(self.adjacency_matrix[agent_id] > 0)
        current_payoff = self.payoff_current_choice(agent_id, num_stick_to_traditional, num_adopt_new_tech, N)

        payoff_new = self.gamma_values[agent_id] * N - self.p * num_stick_to_traditional + self.incentives[agent_id]
        payoff_traditional = self.Vl * N - self.p * num_adopt_new_tech

        beta = self.beta

        if self.current_strategies[agent_id] == "Stick to Traditional":
            sigmoid_input = (payoff_new - current_payoff) * beta
            transition_prob = round(1 / (1 + np.exp(-sigmoid_input)),4)

            self.transition_probs.append(transition_prob)
            self.sigmoid_inputs.append(sigmoid_input)

            if random.random() < transition_prob:
                self.current_strategies[agent_id] = "Adopt New Technology"
                if self.incentives[agent_id] == 0:
                    self.spillovers += 1
                else:
                    self.number_of_incentivised += 1
        elif self.current_strategies[agent_id] == "Adopt New Technology":
             sigmoid_input = (payoff_traditional- current_payoff) * beta
             transition_prob = round(1 / (1 + np.exp(-sigmoid_input)),4)
             if random.random() < transition_prob:
                self.current_strategies[agent_id] = "Stick to Traditional"

    def step(self, max_steps = 15):
        """
        Execute one step of the model.
        :param int max_steps: Maximum number of steps to run the model for.
        """
        np.random.seed(self.seed)
        unchanged_steps = 0

        for step_count in range(max_steps):

            new_tech_count = 0

            total = self.total_to_distribute
            print("total to distribute: " + str(total))

            initial_strategies = np.copy(self.current_strategies)

            for _ in range(self.num_agents):

                if((np.count_nonzero(self.current_strategies == "Adopt New Technology") / 1000) >= 0.95):
                    pass
                else:
                    self.timesteps_95 += 1

                agent_id = random.randint(0, self.num_agents - 1)
                neighbors = np.nonzero(self.adjacency_matrix[agent_id])[1]

                num_stick_to_traditional = np.sum(self.current_strategies[neighbors] == "Stick to Traditional")
                num_adopt_new_tech = np.sum(self.current_strategies[neighbors] == "Adopt New Technology")

                N = num_stick_to_traditional + num_adopt_new_tech

                if self.current_strategies[agent_id] == "Adopt New Technology":
                    new_tech_count += 1

                payoff_new = self.gamma_values[agent_id] * N - self.p * num_stick_to_traditional
                payoff_traditional = self.Vl * N - self.p * num_adopt_new_tech

                # Maximum incentive to give -> add a bit more so that they are not equal
                payoff_needed_for_change = payoff_traditional - payoff_new + 1

                if ((total - payoff_needed_for_change) > 0) and (self.current_strategies[agent_id] == "Stick to Traditional") and (payoff_needed_for_change > 0) and any(self.current_strategies != "Adopt New Technology"):
                    if random.random() <= self.probabilities_informed[agent_id]:
                        self.incentives[agent_id] = payoff_needed_for_change
                        self.total_incentives_ot[agent_id] += payoff_needed_for_change
                        self.update_strategy(agent_id, num_stick_to_traditional, num_adopt_new_tech)
                        self.incentive_amounts.append(payoff_needed_for_change)
                        total -= payoff_needed_for_change
                    else:
                        self.incentives[agent_id] = 0
                        self.update_strategy(agent_id, num_stick_to_traditional, num_adopt_new_tech)
                else:
                    self.incentives[agent_id] = 0
                    self.update_strategy(agent_id, num_stick_to_traditional, num_adopt_new_tech)

            # Calculate the % of norm abandonment
            pct_norm_abandonmnet = (new_tech_count / self.num_agents)
            self.pct_norm_abandonmnet.append(pct_norm_abandonmnet)

            self.total_to_distribute -= (self.total_to_distribute - total)

            # Check if in this timeunit, anything has changed. If there are three timeunits with no changes, the simulation can end.
            if np.array_equal(initial_strategies, self.current_strategies) :
                unchanged_steps += 1
            else:
                unchanged_steps = 0

            if unchanged_steps == 3:
                remaining_steps = max_steps - step_count - 1
                self.pct_norm_abandonmnet.extend([pct_norm_abandonmnet] * remaining_steps)
                break