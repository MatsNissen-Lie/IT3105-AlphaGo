import random

import numpy as np
from MCTS import Node
from game.game_interface import GameInterface


class TreePlolicy:

    def __init__(self, node: Node):
        self.node = node
        self.c = 1.4

    def UCT(self, node) -> float:
        """
        Calculate the value of the child node.

        Parameters
        ----------
        child_node: Node
            The child node.

        Returns
        -------
        value: float
            The value of the child node.
        """
        epsilon = 1

        if node.visits == 0:
            q_value = 0
        else:
            q_value = node.value / node.visits
        exploration_bonus = self.c * np.sqrt(
            np.log(self.node.visits + epsilon) / (node.visits + epsilon)
        )
        return (
            q_value + exploration_bonus
            if self.node.state.get_player() == 1
            else q_value - exploration_bonus
        )

    def maximize(self):
        """
        Select the child node with the highest value.

        Returns
        -------
        max_child_node: Node
            The child node with the highest value.
        """
        max_child_node = None
        max_value = float("-inf")
        for child in self.node.children:
            if child.value > max_value:
                max_value = child.value
                max_child_node = child
        return max_child_node


class DefaultPolicy:
    """
    The DefaultPolicy class is used to represent the default policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to evaluate the leaf nodes. As the default policy, the target policy is
    used, since we are using on-policy Monte Carlo Tree Search.
    """

    def __call__(self, curr_node: Node) -> GameInterface:
        """
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        curr_state : state of the leaf node

        return
        ------
        curr_state : terminal state of the game
        """

        # TODO: The default policy should not add nodes to the tree. It should only simulate the game.
        curr_state = curr_node.state
        while not curr_state.is_terminal():
            possible_moves = curr_state.get_legal_moves()
            move = random.choice(possible_moves)
            curr_state = curr_state.make_move(move)
        return curr_state


class TargetPolicy:
    """
    The TargetPolicy class is used to represent the target policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to select the best child node to explore. As the target policy, the actor
    is used, since we are using on-policy Monte Carlo Tree Search.
    """

    # TODO: The target policy should NOT add nodes to the tree.
    def __call__(self, curr_node: Node, epsilon: float = 0.1) -> GameInterface:
        """
        Using the actor to select the best child node to explore. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        curr_state : state of the leaf node
        possible_moves : list of possible moves

        return
        ------
        move : best move to explore
        """

        while not curr_node.state.is_terminal():
            if random.random() < epsilon:
                possible_moves = curr_node.state.get_legal_moves()
                move = random.choice(possible_moves)
            else:
                move = self.NN.predict_best_move(curr_node.state)
                # state_representation
        return move
