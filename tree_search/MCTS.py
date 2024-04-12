import copy
import random
from typing import Optional, Sequence

if __name__ == "__main__":
    import sys
    from os.path import dirname

    root_path = dirname(sys.path[0])
    print("root_path:", root_path)
    sys.path.append(root_path)

from game.game_interface import GameInterface
from game.nim import Nim

from tree_search.policy import DefaultPolicy, TreePlolicy
from typing import List

from PrettyPrint import PrettyPrintTree

print("en gang ")


class Node:
    def __init__(self, state: GameInterface, parent=None):
        self.state = state
        self.parent: Node | None = parent
        self.children: List["Node"] = []
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

    def __str__(self):
        # pices on board
        pieces = self.state.get_state()
        # if piceces is a list count the number of pices
        if isinstance(pieces, list):
            num = 0
            for i in pieces.flatten():
                if i != 0:
                    num += 1
        return f"Node: {pieces} Visits: {self.visits} Value: {self.value}"


class MCTS:
    def __init__(self, neural_net, iteration_limit, game: GameInterface, M=500):
        self.root = Node(game)
        self.tree_policy = TreePlolicy()
        self.iteration_limit = iteration_limit
        self.M = M  # Number of rollouts
        self.NN = neural_net
        self.NN_confidence = 0.3  # Starting value
        self.game = game
        self.player = game.get_player()
        # self.tree_policy = TreePlolicy(self.NN_confidence)

    def select_node(self):
        return self.tree_policy.search(self.root)

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
        num_rollouts = 10

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
        isRandom (bool): Whether to use a random policy for the rollout.
        Returns:
        float: The estimated value of the node.
        """
        root_node: Node = copy.deepcopy(node)
        simulation = DefaultPolicy()
        terminal_state = simulation(root_node)
        winner = terminal_state.check_win()

        # we minimize for player 2 and maximize for player 1
        return winner if winner != 2 else -1

    def backpropagate(self, node: Node, value):
        while node is not None:
            node.update(value)
            # print("state:",node.state.state)
            # print("player turn: ",node.state.player_turn)
            # print("value:",node.value)
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
        return self.best_moves()

    def best_moves(self):
        # Implement logic to choose the best move from the root node
        # get children from root node and order them by visits
        moves = []
        for child in self.root.children:
            moves.append((child.state.state, child.visits))
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


mcts = MCTS(None, 2, Nim(9, 8))
res = mcts.run(mcts.game)

print(res)
pt = PrettyPrintTree(lambda x: x.children, lambda x: x.value)
pt(mcts.root)
