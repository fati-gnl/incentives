"""
model.py

This file defines the GameModel class, a Mesa model representing the dynamics of agents in a networked game scenario.
Agents interact with their neighbors, updating their strategies based on payoff calculations and individual thresholds.

Methods:
    - generate_truncated_normal: Generate Vh values from a truncated normal distribution.
    - get_network_with_colors: Return the network with node colors based on agents' strategies.
    - step: Execute one step of the model, updating agent strategies and collecting data.
    - get_final_cascade_size_scaled: Get the final cascade size considering only non-initiator agents.
    - get_final_cascade_size: Get the final cascade size considering all agents.
"""
import networkx as nx
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from scipy.stats import truncnorm
from agent_2 import GameAgent
from mesa.datacollection import DataCollector
import random
from numba import njit

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

class GameModel(Model):
    def __init__(self, num_agents, network, Vl, p, total_to_distribute, seed, incentive_strategy):
        """
        Initialize a GameModel.
        :param int num_agents: Number of agents in the model.
        :param network: Parameters for generating the network (e.g., size, connectivity).
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        :param int total_to_distribute: Amount of incentive to distribute
        :param String incentive_strategy: Strategy for selecting who will receive the incentive: "Random", "Highest_degree", "Lowest_degree", "Highest_gamma", "Lowest_gamma"
        """
        self.num_agents = num_agents
        self.network = network
        self.schedule = RandomActivation(self)
        self.pos = nx.spring_layout(self.network)
        self.seed = seed
        self.total_to_distribute = total_to_distribute
        self.incentive_strategy = incentive_strategy
        self.p = p
        self.Vl = Vl

        # Get the gamma values and initial strategies from the network
        gamma_values = nx.get_node_attributes(self.network, 'gamma')
        initial_strategies = nx.get_node_attributes(self.network, 'strategy')

        # Initialise each node with an alpha value and an strategy
        for node in self.network.nodes:
            initial_strategy = initial_strategies[node]
            agent = GameAgent(node, self, Vl, p, initial_strategy, gamma_values)
            self.schedule.add(agent)

        # Add agents to the network
        for i, j in self.network.edges():
            self.schedule.agents[i].neighbors.append(self.schedule.agents[j])
            self.schedule.agents[j].neighbors.append(self.schedule.agents[i])

        self.pct_norm_abandonmnet = []

        self.agent_dict = {agent.unique_id: agent for agent in self.schedule.agents}

        # DataCollector to track strategy changes
        #self.datacollector = DataCollector(
            #agent_reporters={"Strategy": "strategy", "Identifier": "identifier"}
        #)

        self.datacollector = DataCollector(
            agent_reporters={"Incentive": "incentive", "Strategy": "strategy"},
        )

    @staticmethod
    def generate_truncated_normal(mean: float, lower_bound:float, upper_bound:float, size: int) -> np.ndarray:
        """
        Generate values from a truncated normal distribution.
        :param int mean: Mean value for the distribution.
        :param int lower_bound: Lower bound for the truncated distribution.
        :param int upper_bound: Upper bound for the truncated distribution.
        :param int size: Number of values to generate.
        :return: Generated values from the truncated normal distribution.
        """
        a, b = (lower_bound - mean) / (upper_bound - mean), (upper_bound - mean) / (upper_bound - mean)
        values = truncnorm.rvs(a, b, loc=mean, scale=(upper_bound - mean), size=size)
        return values

    def get_network_with_colors(self):
        """
        Return the network with node colors based on agents' strategies.
        :return: Dictionary containing network graph and node colors.
        """
        color_map = {"Stick to Traditional": "red", "Adopt New Technology": "green"}
        colors = [color_map[agent.strategy] for agent in self.schedule.agents]
        gamma_values = {agent.unique_id: f'{agent.gamma:.2f}' for agent in self.schedule.agents}
        return {"graph": self.network, "pos": self.pos, "colors": colors, "gamma_values": gamma_values}

    def step(self):
        """
        Execute one step of the model.
        """
        N = len(self.schedule.agents)
        new_tech_count = 0
        updated_agents = []

        if self.schedule.steps == 0:
            # Get the sorted by by incentive sorting strategy
            sorted_nodes = sort_by_incentive_dist(self.network, self.seed, self.incentive_strategy)
            total = self.total_to_distribute
            for node in sorted_nodes:
                # If there is still incentive to distribute
                if total > 0:
                    agent = self.agent_dict.get(node)
                    # Get the number of neighbours with each strategy
                    num_stick_to_traditional = np.sum([neighbor.strategy == "Stick to Traditional" for neighbor in agent.neighbors])
                    num_adopt_new_tech = np.sum([neighbor.strategy == "Adopt New Technology" for neighbor in agent.neighbors])

                    agent.num_traditional = num_stick_to_traditional
                    agent.num_new = num_adopt_new_tech

                    if agent.strategy == "Adopt New Technology":
                        new_tech_count += 1

                    # Total number of neighbours
                    N = num_stick_to_traditional + num_adopt_new_tech
                    # Calculate initial payoffs
                    payoff_new = agent.gamma * N - self.p * num_stick_to_traditional
                    payoff_traditional = self.Vl * N - self.p * num_adopt_new_tech
                    # Maximum incentive to give
                    payoff_needed_for_change = max(0, payoff_traditional - payoff_new)

                    if total - payoff_needed_for_change > 0:
                        agent.incentive = payoff_needed_for_change
                        agent.update_strategy(agent.num_traditional, agent.num_new)
                        updated_agents.append(agent)
                        total -= payoff_needed_for_change
                    else:
                        agent.incentive = total
                        agent.update_strategy(agent.num_traditional, agent.num_new)
                        updated_agents.append(agent)
                        total = 0
                else:
                    break
            self.datacollector.collect(self)

        # if self.schedule.steps == 0:
        #     # First timestep: Distribute the total incentive equally among all agents
        #     incentive_per_agent = self.total_to_distribute
        #     for agent in self.schedule.agents:
        #         agent.incentive = incentive_per_agent
        # elif self.schedule.steps == 1:
        #     # Second timestep: Set the incentive of all agents to 0
        #     for agent in self.schedule.agents:
        #         agent.incentive = 0

        agents_to_update = [agent for agent in self.schedule.agents if agent not in updated_agents]
        for agent in agents_to_update:
        #for agent in self.schedule.agents:
            # Count the number of neighbours with each strategy
            num_stick_to_traditional = np.sum([neighbor.strategy == "Stick to Traditional" for neighbor in agent.neighbors])
            num_adopt_new_tech = np.sum([neighbor.strategy == "Adopt New Technology" for neighbor in agent.neighbors])

            agent.num_traditional = num_stick_to_traditional
            agent.num_new = num_adopt_new_tech

            if agent.strategy == "Adopt New Technology":
                new_tech_count += 1

            agent.update_strategy(agent.num_traditional, agent.num_new)

        # Calculate the % of norm abandonment
        pct_norm_abandonmnet = (new_tech_count / N) * 100
        self.pct_norm_abandonmnet.append(pct_norm_abandonmnet)

        self.schedule.step()

    def get_final_cascade_size_scaled(self):
        """
        Get the final cascade size considering only non-initiator agents.
        :return: cascade size
        """
        # Get all recorded data
        all_data = self.datacollector.get_agent_vars_dataframe()

        # Identify agents whose strategy changed to "Adopt New Technology" at least once
        changed_agents = all_data[
            (all_data["Strategy"] == "Adopt New Technology") & (all_data["Initiator"] == False)
            ]

        unique_changed_agents = changed_agents.drop_duplicates(subset=["Identifier"])

        # Calculate cascade size (number of unique agents)
        cascade_size = len(unique_changed_agents)
        return cascade_size

    def get_final_cascade_size(self):
        """
        Get the final cascade size considering all agents.
        :return: cascade size
        """
        # Get all recorded data
        all_data = self.datacollector.get_agent_vars_dataframe()

        # Identify agents whose strategy changed to "Adopt New Technology" at least once
        changed_agents = all_data[
            (all_data["Strategy"] == "Adopt New Technology")
            ]

        unique_changed_agents = changed_agents.drop_duplicates(subset=["Identifier"])

        # Calculate cascade size (number of unique agents)
        cascade_size = len(unique_changed_agents)
        return cascade_size

    def calculate_lorenz_curve(self):
        """
        Calculate the Lorenz curve for the current distribution of incentives.
        """
        incentives = self.datacollector.get_agent_vars_dataframe()["Incentive"].values

        sorted_incentives = np.sort(incentives)

        cumulative_percentage = np.cumsum(sorted_incentives) / np.sum(sorted_incentives)

        return cumulative_percentage