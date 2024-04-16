from abc import ABC, abstractmethod
from typing import List, Literal, Tuple


class GameInterface(ABC):
    # self.board = np.zeros((board_size, board_size))
    @abstractmethod
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of all legal moves from the given state.
        """
        pass

    @abstractmethod
    def get_player(self, state) -> Literal[1, 2]:
        """
        Return the current player for the given state.
        """
        pass

    @abstractmethod
    def make_move(self, move: Tuple[int, int]):
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
    def clone(self) -> "GameInterface":
        """
        Return a deep copy of the game.
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current state of the game.
        """
        pass

    @abstractmethod
    def draw_state(self):
        """
        Draw the current state of the game.
        """
        pass

    @abstractmethod
    def get_move_from_nn_output(self, pred):
        """
        Get the move from the neural network output.
        """
        pass
