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
from mesa import Model
from mesa.time import RandomActivation
from scipy.stats import truncnorm
from src.agent import GameAgent
from mesa.datacollection import DataCollector

class GameModel(Model):
    def __init__(self, num_agents, network, node_degrees, Vl, p):
        """
        Initialize a GameModel.
        :param int num_agents: Number of agents in the model.
        :param network: Parameters for generating the network (e.g., size, connectivity).
        :param node_degrees: The distinct degrees of the network.
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        """
        self.num_agents = num_agents
        self.network = network
        self.node_degrees = node_degrees
        self.schedule = RandomActivation(self)
        self.pos = nx.spring_layout(self.network)

        # Store node degrees
        self.node_degrees = nx.degree(network)

        # Get the gamma values and initial strategies from the network
        gamma_values = nx.get_node_attributes(self.network, 'gamma')
        initial_strategies = nx.get_node_attributes(self.network, 'strategy')

        # Initialise each node with an alpha value and an strategy
        for node, degree in node_degrees.items():
            initial_strategy = initial_strategies[node]
            if initial_strategy == "Adopt New Technology":
                initiator = True
            else:
                initiator = False
            agent = GameAgent(node, self, Vl, p, initial_strategy, gamma_values, initiator)
            self.schedule.add(agent)

        # Add agents to the network
        for i, j in self.network.edges():
            self.schedule.agents[i].neighbors.append(self.schedule.agents[j])
            self.schedule.agents[j].neighbors.append(self.schedule.agents[i])

        self.pct_norm_abandonmnet = []

        # DataCollector to track strategy changes
        self.datacollector = DataCollector(
            agent_reporters={"Strategy": "strategy", "Initiator": "initiator", "Identifier": "identifier"}
        )

    @staticmethod
    def generate_truncated_normal(mean, lower_bound, upper_bound, size=1):
        """
        Generate values from a truncated normal distribution.
        :param mean: Mean value for the distribution.
        :param lower_bound: Lower bound for the truncated distribution.
        :param upper_bound: Upper bound for the truncated distribution.
        :param size: Number of values to generate.
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
        alpha_values = {agent.unique_id: f'{agent.alpha:.2f}' for agent in self.schedule.agents}
        return {"graph": self.network, "pos": self.pos, "colors": colors, "alpha_values": alpha_values}

    def step(self):
        """
        Execute one step of the model.
        """
        N = len(self.schedule.agents)
        new_tech_count = 0

        for agent in self.schedule.agents:
            # Count the number of neighbours with each strategy
            num_stick_to_traditional = sum(
                1 for neighbor in agent.neighbors if neighbor.strategy == "Stick to Traditional")
            num_adopt_new_tech = sum(1 for neighbor in agent.neighbors if neighbor.strategy == "Adopt New Technology")

            agent.num_traditional = num_stick_to_traditional
            agent.num_new = num_adopt_new_tech

            if agent.strategy == "Adopt New Technology":
                new_tech_count += 1

        # Calculate the % of norm abandonment
        pct_norm_abandonmnet = (new_tech_count / N) * 100
        self.pct_norm_abandonmnet.append(pct_norm_abandonmnet)

        self.datacollector.collect(self)

        for agent in self.schedule.agents:
            agent.update_strategy(agent.num_traditional, agent.num_new)

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