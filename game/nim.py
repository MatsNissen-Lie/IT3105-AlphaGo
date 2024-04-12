from abc import ABC, abstractmethod

from game.game_interface import GameInterface
import numpy as np


class Nim(GameInterface):
    def __init__(self, initial_state: int, max_take: int):
        self.state = initial_state  # initial_state should be a list of integers representing piles
        self.max_take = max_take
        self.player_turn = 1

    def get_state(self):
        return self.state

    def get_legal_moves(self):
        """
        Generate all legal moves within the constraints of the maximum number of objects
        that can be taken in one move.
        """
        n = min(self.max_take, self.state)
        moves = []
        for i in range(1, n + 1):
            moves.append(i)
        return moves

    def get_player(self):
        return self.player_turn

    def is_valid_move(self, move):
        """
        Check if a move is valid given the current state.
        """
        return (move >= 1) and (move <= self.max_take) and (self.max_take - move >= 0)

    def make_move(self, move):
        """
        Apply a move to the state. A move is a number of objects to take from the pile.
        This function returns the new state after the move.
        """
        if self.is_valid_move(move):
            self.state -= move
            self.player_turn = 3 - self.player_turn  # Switch player
            return self
        else:
            raise Exception()

    def print_pile(self):
        # print the piles nicely
        print(self.state)

    def is_terminal(self):
        """
        The game is over when all piles are empty.
        """
        return self.state <= 0

    def check_win(self):
        """
        The game is over when all piles are empty.
        Returns 1 if player 1 wins, 2 if player 2 wins, and 0 if it's a draw.
        """
        if not self.is_terminal():
            return None
        return self.player_turn

    def get_nn_input(self):
        """
        Prepare the state for neural network input. This can vary depending on the NN architecture.
        For simplicity, let's just return the state as is.
        """
        return self.state

    def get_state(self):
        return self.state

    def clone(self):
        clone = Nim(self.state, self.max_take)
        clone.player_turn = int(self.player_turn)
        return clone


if __name__ == "__main__":
    # Test the Nim class
    game = Nim(10, 3)
    game.print_pile()
    # moves
    print(game.get_legal_moves())

    starting_player = game.player_turn
    print("game.player_turn", starting_player)
    game.make_move(3)  # player 1
    assert game.player_turn == 2
    game.make_move(3)  # player 2
    assert starting_player == game.player_turn
    assert game.get_state() == 4
    moves = game.make_move(3)  # player 1
    assert len(game.get_legal_moves()) == 1
    print("last move by", game.player_turn)
    game.make_move(1)  # palyer 2
    print("winner", game.check_win())
    assert game.is_terminal()

    assert game.check_win() == 1
