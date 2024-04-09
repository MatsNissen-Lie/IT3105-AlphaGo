from abc import ABC, abstractmethod

from game.game_interface import GameInterface


class Nim(GameInterface):
    def __init__(self, initial_state, max_take=3):
        self.state = initial_state  # initial_state should be a list of integers representing piles
        self.max_take = max_take

    def get_legal_moves(self, state):
        """
        Generate all legal moves within the constraints of the maximum number of objects
        that can be taken in one move.
        """
        legal_moves = []
        for pile_index, pile_size in enumerate(state):
            for objects_to_remove in range(1, min(pile_size, self.max_take) + 1):
                legal_moves.append((pile_index, objects_to_remove))
        return legal_moves

    def make_move(self, state, move):
        """
        Apply a move to the state. A move is a tuple (pile_index, objects_to_remove).
        This function returns the new state after the move.
        """
        pile_index, objects_to_remove = move
        new_state = state[:]
        new_state[pile_index] -= objects_to_remove
        return new_state

    def is_terminal(self, state):
        """
        The game is over when all piles are empty.
        """
        return all(pile == 0 for pile in state)

    def get_nn_input(self, state):
        """
        Prepare the state for neural network input. This can vary depending on the NN architecture.
        For simplicity, let's just return the state as is.
        """
        return state


# Example usage
initial_state = [3, 4, 5]  # Three piles with 3, 4, and 5 objects respectively
nim_game = NimStateManager(initial_state)
print("Legal moves from initial state:", nim_game.get_legal_moves(nim_game.state))
new_state = nim_game.make_move(
    nim_game.state, (0, 2)
)  # Take 2 objects from the first pile
print("New state after move:", new_state)
print("Is game over?", nim_game.is_terminal(new_state))
