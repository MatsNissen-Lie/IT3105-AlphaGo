from abc import ABC, abstractmethod

from game.game_interface import GameInterface


class Nim(GameInterface):
    def __init__(self, initial_state, max_take=3):
        self.state = initial_state  # initial_state should be a list of integers representing piles
        self.max_take = max_take
        self.player_turn = 1

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

    def is_valid_move(self, move):
        """
        Check if a move is valid given the current state.
        """
        pile_index, objects_to_remove = move
        if pile_index < 0 or pile_index >= len(self.state):
            return False
        if objects_to_remove < 1 or objects_to_remove > self.max_take:
            return False
        if objects_to_remove > self.state[pile_index]:
            return False
        return True

    def make_move(self, state, move):
        """
        Apply a move to the state. A move is a tuple (pile_index, objects_to_remove).
        This function returns the new state after the move.
        """
        pile_index, objects_to_remove = move
        new_state = state[:]
        new_state[pile_index] -= objects_to_remove
        self.player_turn = 3 - self.player_turn  # Switch player
        return new_state

    def print_piles(self):
        # print the piles nicely
        pile_string = ""
        for i, pile in enumerate(self.state):
            pile_string += f"Pile {i}: {pile} objects\n"
        print(pile_string)

    def is_terminal(self):
        """
        The game is over when all piles are empty.
        """
        return all(pile == 0 for pile in self.state)

    def check_win(self):
        if not self.is_terminal():
            return None
        return 3 - self.player_turn

    def get_nn_input(self, state):
        """
        Prepare the state for neural network input. This can vary depending on the NN architecture.
        For simplicity, let's just return the state as is.
        """
        return state


# Example usage
initial_state = [3, 4, 5]  # Three piles with 3, 4, and 5 objects respectively
nim_game = Nim(initial_state)
print("Legal moves from initial state:", nim_game.get_legal_moves(nim_game.state))
new_state = nim_game.make_move(
    nim_game.state, (0, 2)
)  # Take 2 objects from the first pile
print("New state after move:", new_state)
print("Is game over?", nim_game.is_terminal(new_state))
