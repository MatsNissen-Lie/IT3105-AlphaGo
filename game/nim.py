from abc import ABC, abstractmethod

from game.game_interface import GameInterface


class Nim(GameInterface):
    def __init__(self, initial_state, max_take):
        self.state = initial_state  # initial_state should be a list of integers representing piles
        self.max_take = max_take
        self.player_turn = 1

    def get_legal_moves(self):
        """
        Generate all legal moves within the constraints of the maximum number of objects
        that can be taken in one move.
        """
        moves = []
        for i in range(1, self.max_take+1):
            moves.append(i)
        return moves

    def is_valid_move(self, move):
        """
        Check if a move is valid given the current state.
        """
        return ((move >= 1) and (move <= self.max_take) and (self.max_take-move >= 0))


    def make_move(self, move):
        """
        Apply a move to the state. A move is a tuple (pile_index, objects_to_remove).
        This function returns the new state after the move.
        """
        if self.is_valid_move(move):
            
            self.player_turn = 3 - self.player_turn  # Switch player
            self.state -= move 
            return self
        else:
            raise Exception()

    def print_piles(self):
        # print the piles nicely
        print(self.state)

    def is_terminal(self):
        """
        The game is over when all piles are empty.
        """
        return self.state <= 0

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

    def get_state(self):
        return self.state


# # Example usage
# initial_state = [3, 4, 5]  # Three piles with 3, 4, and 5 objects respectively
# nim_game = Nim(initial_state)
# print("Legal moves from initial state:", nim_game.get_legal_moves())
# new_state = nim_game.make_move(
#     nim_game.state, (0, 2)
# )  # Take 2 objects from the first pile
# print("New state after move:", new_state)
# print("Is game over?", nim_game.is_terminal(new_state))
