from mesa import Agent
import random

class GameAgent(Agent):
    def __init__(self, unique_id, model, Vl, Vh, p, initial_strategy_prob):
        """
        Initialize a game agent
        :param int unique_id: Unique identifier for the agent.
        :param GameModel model: Reference to the model containing the agent.
        :param float Vl: Low reward for not selecting their preferred strategy.
        :param float Vh: High reward for selecting their preferred strategy.
        :param float p: Penalty for miscoordination with neighbours.
        :param initial_strategy_prob: Probability for initially selecting the "Adopt New Technology" strategy.
        """
        super().__init__(unique_id, model)
        self.strategy = "Stick to Traditional"
        self.Vl = Vl
        self.Vh = Vh
        self.p = p
        # TODO: Añadir como parametros si lo haremos luego con eso
        self.alpha = random.choice(model.generate_truncated_normal(mean=15, lower_bound=13, upper_bound=17, size=1000))
        self.neighbors = []
        self.num_traditional = 0
        self.num_new = 0

        #self.strategy = "Adopt New Technology" if random.random() < initial_strategy_prob else "Stick to Traditional"

    def individual_threshold(self, num_stick_to_traditional, num_adopt_new_tech):
        """
        Calculate the individual's switching threshold for strategy update.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        :return: Individual threshold value.
        """
        N = len(self.neighbors)
        individual_threshold_value = N * (self.alpha - self.Vl - self.p * ((num_stick_to_traditional - num_adopt_new_tech)/ N))
        return individual_threshold_value

    def payoff_current_choice(self, num_stick_to_traditional, num_adopt_new_tech):
        """
        Calculate the payoff for playing the current agents strategy.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        :return: Payoff for the current strategy
        """
        N = len(self.neighbors)
        if self.strategy == "Adopt New Technology":
            payoff = self.alpha * N - self.p * num_stick_to_traditional
        else:
            payoff = self.Vl * N - self.p * num_adopt_new_tech
        return payoff

    def update_strategy(self, num_stick_to_traditional, num_adopt_new_tech):
        """
        Update the agent's strategy based on the calculated payoff and individual threshold.
        :param int num_stick_to_traditional: Number of neighbors sticking to the traditional strategy.
        :param int num_adopt_new_tech: Number of neighbors who have adopted the new technology strategy.
        """
        proportion_deviated = num_adopt_new_tech / len(self.neighbors)
        self.individual_threshold_value = self.individual_threshold(num_stick_to_traditional, num_adopt_new_tech)
        current_payoff = self.payoff_current_choice(num_stick_to_traditional, num_adopt_new_tech)

        # Calculate the payoffs for their opposite choice
        N = len(self.neighbors)
        payoff_new = self.Vl * N - self.p * num_adopt_new_tech
        payoff_traditional = self.alpha * N - self.p * num_stick_to_traditional

        if current_payoff < payoff_traditional and self.strategy == "Stick to Traditional":
            self.strategy = "Adopt New Technology"
        elif current_payoff < payoff_new and self.strategy == "Adopt New Technology":
            self.strategy = "Stick to Traditional"

        #if payoff < self.individual_threshold_value and self.strategy == "Stick to Traditional":
            #self.strategy = "Adopt New Technology"
            #print("has entered")
        #elif payoff < self.individual_threshold_value and self.strategy == "Adopt New Technology":
            #self.strategy = "Stick to Traditional"

        # Compare with individual threshold and update strategy accordingly
        #if self.individual_threshold_value > proportion_deviated:
            #self.strategy = "Adopt New Technology"
        #else:
            #self.strategy = "Stick to Traditional"