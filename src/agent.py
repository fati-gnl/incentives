"""
agent.py

This file defines the GameAgent class, which represents an agent in the simulation.
Agents have two possible strategies ("Stick to Traditional" or "Adopt New Technology"),
and they update their strategies based on payoff calculations and individual thresholds.

Methods:
    - payoff_current_choice: Calculate the payoff for the current strategy.
    - update_strategy: Update the agent's strategy based on calculated payoff and individual threshold.
"""

from mesa import Agent

class GameAgent(Agent):
    def __init__(self, unique_id, model, Vl, p, initial_strategy, gamma_values, initiator):
        """
        Initialize a game agent
        :param int unique_id: Unique identifier for the agent.
        :param GameModel model: Reference to the model containing the agent.
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        :param string initial_strategy: Either "Stick to Traditional" or "Adopt New Technology".
        :param gamma_values: The list of alpha values.
        :param Boolean initiator: Whether or not their strategy is "Adopt New Technology" at the start of the simulation
        """
        super().__init__(unique_id, model)
        self.strategy = initial_strategy
        self.identifier = unique_id
        self.Vl = Vl
        self.p = p
        self.gamma = gamma_values[unique_id]
        self.neighbors = []
        self.num_traditional = 0
        self.num_new = 0
        self.initiator = initiator

    def payoff_current_choice(self, num_stick_to_traditional, num_adopt_new_tech):
        """
        Calculate the payoff for playing the current agents strategy.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        :return: Payoff for the current strategy
        """
        N = len(self.neighbors)
        if self.strategy == "Adopt New Technology":
            payoff = self.gamma * N - self.p * num_stick_to_traditional
        else:
            payoff = self.Vl * N - self.p * num_adopt_new_tech
        return payoff

    def update_strategy(self, num_stick_to_traditional, num_adopt_new_tech):
        """
        Update the agent's strategy based on the calculated payoff and individual threshold.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        """
        current_payoff = self.payoff_current_choice(num_stick_to_traditional, num_adopt_new_tech)

        # Calculate the payoffs for their opposite choice
        N = len(self.neighbors)
        payoff_new = self.gamma * N - self.p * num_stick_to_traditional
        payoff_traditional = self.Vl * N - self.p * num_adopt_new_tech

        if current_payoff < payoff_new and self.strategy == "Stick to Traditional":
            self.strategy = "Adopt New Technology"
        elif current_payoff < payoff_traditional and self.strategy == "Adopt New Technology":
            self.strategy = "Stick to Traditional"