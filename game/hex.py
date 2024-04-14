import copy
from typing import List, Tuple
import numpy as np


class Hex:
    def __init__(self, board_size=7):
        if not 3 <= board_size <= 10:
            raise ValueError("Board size must be between 3 and 10")
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.player_turn = 1
        self.action = None

    def get_player(self):
        return self.player_turn

    def get_previous_action(self):
        return self.action

    def get_state(self):
        return self.board

    def is_valid_move(self, row, col):
        return (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row][col] == 0
        )

    def get_legal_moves(self):
        """
        Return a list of all legal moves from the current state.
        """
        legal_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 0:  # If the cell is empty
                    legal_moves.append((row, col))
        return legal_moves

    def is_terminal(self):
        """
        Check if the game is over for the current state.
        """
        # Check if there is a winner
        winner = self.check_win()
        if winner != 0:
            return True

        # Check if there are any legal moves left
        # Is this necessary? There be a winner before the board is full in hex
        # if not self.get_legal_moves():
        #     print("No legal moves left. This should never happen.")
        #     return True

        return False

    def is_terminal(self):
        return self.check_win() != 0

    def make_move(self, move: Tuple[int, int]):
        row, col = move
        if self.is_valid_move(row, col):
            self.board[row][col] = self.player_turn
            self.player_turn = 3 - self.player_turn  # Switch player
            self.action = move
            return True
        return False

    def get_player_turn(self):
        return self.player_turn

    def check_win(self):
        # For each player, check if there's a winning path
        for player in [1, 2]:
            visited = set()
            if player == 1:
                for col in range(self.board_size):
                    if self.board[0][col] == player and self.dfs(
                        0, col, player, visited
                    ):
                        return player
            else:
                for row in range(self.board_size):
                    if self.board[row][0] == player and self.dfs(
                        row, 0, player, visited
                    ):
                        return player
        return 0  # No winner

    def dfs(self, row, col, player, visited: set):
        if (player == 1 and row == self.board_size - 1) or (
            player == 2 and col == self.board_size - 1
        ):
            return True

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        visited.add((row, col))

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and (r, c) not in visited
                and self.board[r][c] == player
            ):
                if self.dfs(r, c, player, visited):
                    return True

        return False

    def draw_state(self):
        board = self.board
        column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rows, cols, indent = len(board), len(board[0]), 0
        headings = " " * 5 + (" " * 3).join(column_names[:cols])
        tops = " " * 5 + (" " * 3).join("-" * cols)
        roof = " " * 4 + "/ \\" + "_/ \\" * (cols - 1)
        print(headings), print(tops), print(roof)
        color_mapping = lambda i: " WB"[int(i)]
        for r in range(rows):
            row_mid = " " * indent
            row_mid += " {} | ".format(r + 1)
            row_mid += " | ".join(map(color_mapping, board[r]))
            row_mid += " | {} ".format(r + 1)
            print(row_mid)
            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * cols
            if r < rows - 1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " " * (indent - 2) + headings
        print(headings)

    def move_to_str(self, move):
        if move == None:
            return "Pass"
        return f"{chr(move[1] + 65)}{move[0] + 1}"

    def transform_board_values_for_nn(self):
        return np.where(self.board == 1, 1, np.where(self.board == 2, -1, 0))

    def get_nn_input(self):
        flat_board = self.transform_board_for_nn().flatten()
        format_player = 1 if self.player_turn == 1 else -1
        nn_input = np.append(flat_board, format_player)
        return nn_input

    def get_nn_input_advanced(self):
        # Create separate channels for each player
        player1_channel = (self.board == 1).astype(int)
        player2_channel = (self.board == 2).astype(int)

        # Channel for the current player
        current_player_channel = np.full(
            (self.board_size, self.board_size), 1 if self.player_turn == 1 else -1
        )

        # Stack the channels to create a multi-channel input
        stacked_channels = np.stack(
            (player1_channel, player2_channel, current_player_channel)
        )

        # Additional game state information
        num_pieces_player1 = np.sum(player1_channel)
        num_pieces_player2 = np.sum(player2_channel)
        total_moves = num_pieces_player1 + num_pieces_player2

        # Append additional game state information to the channels
        game_state_info = np.array(
            [num_pieces_player1, num_pieces_player2, total_moves]
        )

        # Combine multi-channel input with game state information
        nn_input = (stacked_channels, game_state_info)

        return nn_input

    # make a clone of the board
    def clone(self):
        clone = copy.deepcopy(self)
        return clone

    def transform_nn_output(
        self, move_visits: List[Tuple[Tuple[int, int], int]]
    ) -> np.ndarray:
        """Transform the output of the MCTS into a distribution of visit counts that can be used as targets for traning the neural network.

        Args:
            move_visits (List[Tuple[Tuple[int, int], int]]): A list of moves and their visit counts.

        Returns:
            ndarray[float]: A distribution of visit counts for each move on the board.
        """
        visit_counts = [0] * self.board_size**2
        for move, visits in move_visits:
            index = move[0] * self.board + move[1]
            visit_counts[index] = visits
        total_visit_count = sum(visit_counts)
        distribution = np.array([count / total_visit_count for count in visit_counts])
        return distribution

    def go_to_end_game(self):
        for i in range(self.board_size - 1):
            for j in range(self.board_size):
                self.make_move((i, j))
        self.make_move((6, 6))
        self.make_move((6, 1))


# Uncomment to test the game setup
if __name__ == "__main__":
    game = Hex()
    if False:
        game.make_move((0, 0))  # Player 1
        game.make_move((0, 1))  # Player 2
        game.make_move((1, 0))  # Player 1
        game.make_move((0, 2))  # Player 2
        game.make_move((2, 0))  # Player 1
        game.make_move((0, 3))  # Player 2
        game.make_move((3, 0))  # Player 1
        game.make_move((0, 4))  # Player 2
        game.make_move((4, 0))  # Player 1
        game.make_move((0, 5))  # Player 2
        game.make_move((5, 0))  # Player 1
        game.make_move((0, 6))  # Player 2
        # Player 1, this should create a winning path from top to bottom
        game.make_move((6, 0))
        game.draw_state()
        winner = game.check_win()
        print("Winner:", winner)
    else:
        game.go_to_end_game()
        game.draw_state()
        print(game.check_win())
        print(game.get_player_turn())
        print(game.get_nn_input())
        # print(game.get_nn_input_advanced())
