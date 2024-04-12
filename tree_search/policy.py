import random
from tree_search.node import Node
from game.game_interface import GameInterface
import numpy as np
from typing import Tuple


class TreePlolicy:

    def __init__(self):
        pass

    def get_child(self, node: Node) -> Node:
        """
        Select the child node with the highest UCT value.

        Returns
        -------
        child_node: Node
            The child node with the highest UCT value.
        """
        if node.children == []:
            raise ValueError("The node has no children.")

        optimizer = max if node.get_state().get_player() == 1 else min
        child_node = optimizer(node.children, key=lambda x: x.UCT())
        return child_node

    def search(self, root_node: Node) -> Tuple[Node, int]:
        """
        Select the child node with the highest UCT value.

        Returns
        -------
        child_node: Node
            The child node with the highest UCT value.
        """
        node = root_node
        depth = 1
        while node.children != []:
            node = self.get_child(node)
            depth += 1
        return node, depth


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
        curr_state = curr_node.game_state
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
