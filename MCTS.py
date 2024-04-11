import math
import random
from typing import Optional, Sequence
from game.game_interface import GameInterface
from game.hex import Hex
from game.nim import Nim


class Node:
    def __init__(self, state: GameInterface, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.value = 0  # Value from critic
        self.visits = 0
        self.c = 1.4  # Exploration parameter
        # self.wins = 0  # Number of wins
        # self.loss = 0  # Number of losses

    def is_terminal(self):
        # Implement a method to check if the node is at a terminal state
        return self.state.is_terminal()

    def check_winner(self):
        winner = self.state.check_win()
        return winner if winner != 2 else -1

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.value += result

    def ucb1_score(self, total_parent_visits):
        if self.visits == 0:  # Assign a high score to unvisited nodes for exploration
            return float("inf")
        exploitation = self.value / self.visits
        exploration = math.sqrt(math.log(total_parent_visits) / self.visits)
        # TODO: fix this for player one and two. This is a simple implementation for player 1
        return exploitation + self.c * exploration


class MCTS:
    def __init__(self, neural_net, iteration_limit, game: GameInterface, M=500):
        self.root: Node = None
        self.iteration_limit = iteration_limit
        self.M = M  # Number of rollouts
        self.NN = neural_net
        self.NN_confidence = 0.3  # Starting value
        self.game = game
        self.player = game.get_player()

    def select_node(self):
        """
        This method selects the best node to explore based on the UCB1 formula.
        """
        # kanskje denne metoden skal bruke Anet til å velge hvilken node vi skal gå til. Dette kan være en måte å bruke Anet som en actor i MCTS
        # git det mening å bruke UCB1 her og anet i rollout?
        node = self.root
        while node.children:
            # TODO: fix this for player one and two. This is a simple implementation for player 1
            # max for palyer 1 and min for player 2
            node = max(node.children, key=lambda c: c.ucb1_score(node.visits))
        return node

    def expand(self, node: Node):
        # Get the possible moves from the game state
        possible_moves = self.game.get_legal_moves()
        random.shuffle(possible_moves)
        # TODO: Vi kan bruke Anet til å velge hvilke moves vi skal legge til i treet!
        for move in possible_moves:
            new_state = self.game.clone()
            new_state.make_move(move)
            child_node = Node(state=new_state, parent=node)
            node.children.append(child_node)

    # aka leaf_evaluation
    def simulate(self, node):
        # Use the critic to evaluate the node
        # her kan du gjør en rollout med Anet som actor eller en critic med Anet og spare deg for rollout.
        """
        Perform 500 rollouts from the given node and average their results.

        Args:
        node (Node): The node from which the rollouts start.

        Returns:
        float: The average value of the 500 rollouts.
        """
        total_value = 0
        num_rollouts = 500

        for _ in range(num_rollouts):
            value = self.rollout(node)
            total_value += value

        average_value = total_value / num_rollouts
        return average_value

    def rollout(self, node: Node, epsilon=0.1, isRandom=True):
        """
        Perform a rollout from the given node using an epsilon-greedy strategy.

        Args:
        node (Node): The node from which the rollout starts.
        epsilon (float): The probability of taking a random action.

        Returns:
        float: The estimated value of the node.
        """
        current_state: GameInterface = node.state.clone()
        while not current_state.is_terminal():
            possible_moves = current_state.get_legal_moves()
            if isRandom or random.random() < epsilon:
                # Exploration: choose a random move
                move = random.choice(possible_moves)
            else:
                # Exploitation: choose the best move as suggested by the actor (neural network)
                # TODO: implement NN.predict_best_move() method
                move = self.NN.predict_best_move(current_state, possible_moves)

            current_state = current_state.make_move(move)

        winner = current_state.check_win()

        return 1 if winner != self.player else -1

    def backpropagate(self, node: Node, value):
        while node is not None:
            node.update(value)
            print(node.state.state)
            print(node.state.player_turn)
            print(node.value)
            node = node.parent

    def run(self, initial_state):
        self.root = Node(initial_state)
        for _ in range(self.iteration_limit):
            node = self.select_node()
            if not node.is_terminal():
                self.expand(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
        # Return the best move based on the search
        return self.best_move()

    def best_moves(self):
        # Implement logic to choose the best move from the root node
        # get children from root node and order them by visits
        moves = []
        for child in self.root.children:
            moves.append((child.state, child.visits))
        return moves

    # jeg vil gjøre et approch der vi begynner med å gjøre rollouts også gir vi mer og mer tillit til modellen vår etterhvert som vi har gjort flere rollouts
    def update_critic_confidence(self, critic_accuracy):
        # Adjust the confidence based on the accuracy
        increment = 0.02  # Increment value
        decrement = 0.01  # Decrement value
        threshold = 0.75  # Performance threshold (e.g., 75% accuracy)
        # This is a simplistic adjustment logic; you might want a more sophisticated method
        if critic_accuracy > threshold:  # Define a suitable threshold
            self.NN_confidence = min(self.NN_confidence + increment, 1.0)
        else:
            self.NN_confidence = max(self.NN_confidence - decrement, 0.0)


class Critic:
    def evaluate(self, state):
        # Implement the evaluation logic (e.g., a neural network model)
        return random.random()  # Dummy value for illustration


mcts = MCTS(None, 10, Nim(20, 7))
mcts.game.print_piles()
print(mcts.run(mcts.game))
