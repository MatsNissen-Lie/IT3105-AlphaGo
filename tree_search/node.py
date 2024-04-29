import numpy as np
from config.params import EXPLORATION_CONSTANT
from game.game_interface import GameInterface
from typing import List


class Node:
    def __init__(self, state: GameInterface, parent=None, move=None):
        self.game_state = state
        self.parent: Node | None = parent
        self.children: List["Node"] = []
        self.value = 0  # Value from critic
        self.visits = 0
        self.c = EXPLORATION_CONSTANT

        self.move_from_parent = move  # info only for drawing the tree

    def get_state(self):
        return self.game_state

    def is_terminal(self):
        # Implement a method to check if the node is at a terminal state
        return self.game_state.is_terminal()

    def check_winner(self):
        winner = self.game_state.check_win()
        return winner if winner != 2 else -1

    def add_child(self, child_state, move):
        child_node = Node(child_state, self, move)
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.value += result

    def UCT(self) -> float:
        """
        Calculate the value of the child node. From the parents view.

        Parameters
        ----------
        child_node: Node
            The child node.

        Returns
        -------
        value: float
            The value of the child node.
        """
        if self.parent is None:
            return 0
        node = self
        if node.visits == 0:
            q_value = 0
        else:
            q_value = node.value / node.visits
        exploration_bonus = self.c * np.sqrt(
            # np.log(self.node.visits + epsilon) / (node.visits + epsilon)
            np.log(node.parent.visits)
            / (node.visits + 1)
        )
        total_value = (
            q_value + exploration_bonus
            if node.parent.game_state.get_player() == 1
            else q_value - exploration_bonus
        )
        return total_value

    def __str__(self):
        # pices on board
        pieces = self.game_state.get_state()
        # if piceces is a list count the number of pices
        if isinstance(pieces, list):
            num = 0
            for i in pieces.flatten():
                if i != 0:
                    num += 1
        return f"Node: {pieces} Visits: {self.visits} Value: {self.value}"

    def get_value(self):
        return self.value / max(self.visits, 1)

    def samle_children(self, count=3):
        # get children with most visits, max 3
        return sorted(self.children, key=lambda x: x.visits, reverse=True)[:count]
