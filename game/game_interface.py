from abc import ABC, abstractmethod


class GameInterface(ABC):
    @abstractmethod
    def get_legal_moves(self, state):
        """
        Return a list of all legal moves from the given state.
        """
        pass

    @abstractmethod
    def get_player(self, state):
        """
        Return the current player for the given state.
        """
        pass

    @abstractmethod
    def make_move(self, state, move):
        """
        Apply a move to the state and return the new state.
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        Check if the game is over for the given state.
        """
        pass

    @abstractmethod
    def check_win(self, state):
        """
        Check if the game is over for the given state.
        Returns 1 if player 1 wins, 2 if player 2 wins, and 0 if it's a draw.
        """
        pass

    @abstractmethod
    def get_nn_input(self, state):
        """
        Prepare and return the neural network input for the given state.
        """
        pass

    @abstractmethod
    def clone(self):
        """
        Return a deep copy of the game.
        """
        pass
