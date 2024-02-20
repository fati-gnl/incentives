"""
agent.py

This file defines the GameAgent class, which represents an agent in the simulation.
Agents have two possible strategies ("Stick to Traditional" or "Adopt New Technology"),
and they update their strategies based on payoff calculations and individual thresholds.

Methods:
    - payoff_current_choice: Calculate the payoff for the current strategy.
    - update_strategy: Update the agent's strategy based on calculated payoff and individual threshold.
"""
from numba import njit

class GameAgent:
    def __init__(self, unique_id, Vl, p, initial_strategy, gamma_values):
        """
        Initialize a game agent
        :param int unique_id: Unique identifier for the agent.
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        :param string initial_strategy: Either "Stick to Traditional" or "Adopt New Technology".
        :param gamma_values: The list of alpha values.
        """
        self.unique_id = unique_id
        self.strategy: str = initial_strategy
        self.identifier = unique_id
        self.Vl: float = Vl
        self.p: int = p
        self.gamma: float = gamma_values[unique_id]
        self.neighbors = []
        self.num_traditional: int = 0
        self.c: float = 0
        self.probability_informed = 1

    def payoff_current_choice(self, num_stick_to_traditional: int, num_adopt_new_tech:int, N:int) -> float:
        """
        Calculate the payoff for playing the current agents strategy.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        :return: Payoff for the current strategy
        """
        if self.strategy == "Adopt New Technology":
            payoff = self.gamma * N - self.p * num_stick_to_traditional
        else:
            payoff = self.Vl * N - self.p * num_adopt_new_tech
        return payoff

    def update_strategy(self, num_stick_to_traditional: int, num_adopt_new_tech: int):
        """
        Update the agent's strategy based on the calculated payoff and individual threshold.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        """
        N: int = len(self.neighbors)
        current_payoff = self.payoff_current_choice(num_stick_to_traditional, num_adopt_new_tech, N)

        payoff_new = self.gamma * N - self.p * num_stick_to_traditional + self.incentive
        payoff_traditional = (self.Vl * N - self.p * num_adopt_new_tech)

        if current_payoff <= payoff_new and self.strategy == "Stick to Traditional":
            self.strategy = "Adopt New Technology"
        elif current_payoff < payoff_traditional and self.strategy == "Adopt New Technology":
            self.strategy = "Stick to Traditional"