import math
import random

from typing import Optional, Sequence


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.value = 0  # Value from critic
        self.visits = 0

    def is_terminal(self):
        # Implement a method to check if the node is at a terminal state
        return self.state.is_terminal()

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)

    def update(self, result):
        self.value += (result - self.value) / (self.visits + 1)
        self.visits += 1

    def ucb1_score(self, total_parent_visits):
        if self.visits == 0:  # Assign a high score to unvisited nodes for exploration
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = math.sqrt(math.log(total_parent_visits) / self.visits)
        return exploitation + self.c * exploration


class MCTS:
    def __init__(self, critic, iteration_limit):
        self.root: Node = None
        self.critic = critic
        self.iteration_limit = iteration_limit
        self.critic_confidence = 0.3  # Starting value

    def select_node(self):
        node = self.root
        while node.children:
            log_parent_visits = math.log(node.visits)
            node = max(node.children, key=lambda c: c.ucb1_score(log_parent_visits))
        return node

    def expand(self, node):
        # Get the possible moves from the game state
        possible_moves = self.game.get_possible_moves(node.state)
        random.shuffle(possible_moves)  # Randomize the order of moves
        # For each move, create a new node and add it to the node's children
        # TODO: kanskje vi vil velge move i tilfeldig rekkefølge for å unngå bias
        for move in possible_moves:
            new_state = self.game.make_move(node.state, move)
            child_node = Node(state=new_state, parent=node, move=move)
            node.children.append(child_node)

    def simulate(self, node):
        # Use the critic to evaluate the node
        return self.critic.evaluate(node.state)

    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
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

    def best_move(self):
        # Implement logic to choose the best move from the root node
        pass

    # jeg vil gjøre et approch der vi begynner med å gjøre rollouts også gir vi mer og mer tillit til modellen vår etterhvert som vi har gjort flere rollouts
    def update_critic_confidence(self, critic_accuracy):
        # Adjust the confidence based on the accuracy
        increment = 0.02  # Increment value
        decrement = 0.01  # Decrement value
        threshold = 0.75  # Performance threshold (e.g., 75% accuracy)
        # This is a simplistic adjustment logic; you might want a more sophisticated method
        if critic_accuracy > threshold:  # Define a suitable threshold
            self.critic_confidence = min(self.critic_confidence + increment, 1.0)
        else:
            self.critic_confidence = max(self.critic_confidence - decrement, 0.0)


class Critic:
    def evaluate(self, state):
        # Implement the evaluation logic (e.g., a neural network model)
        pass
