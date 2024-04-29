import copy
from typing import List, Tuple
import numpy as np
from colorama import Back, Style, Fore

# from tree_search.node import Node


class Hex:
    def __init__(self, board_size=7, starting_player=1):

        if not 3 <= board_size <= 10:
            raise ValueError("Board size must be between 3 and 10")
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.player_turn = starting_player
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

    def isAvailable_move(self, row, col):
        return self.board[row][col] == 0

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

    def get_palyer_color(self, cell_value):
        if int(cell_value) == 1:
            cell_display = f"{Back.BLUE} {Style.RESET_ALL}"
        elif int(cell_value) == 2:
            cell_display = f"{Back.RED} {Style.RESET_ALL}"
        else:
            cell_display = " "
        return cell_display

    def draw_state(self, preds=None):
        def color_mapping(cell_value, index=None):
            if int(cell_value) == 1 or int(cell_value) == 2:
                cell_display = self.get_palyer_color(cell_value)
            else:
                if preds is not None and index is not None:
                    prediction = int(round(preds[index] * 10))
                    cell_display = f"{prediction}"
                else:
                    cell_display = " "
            return cell_display

        if preds is not None:
            print(f"Player to move: {color_mapping(self.player_turn, 0)}")
        board = self.board
        column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rows, cols, indent = len(board), len(board[0]), 0
        headings = " " * 5 + (" " * 3).join(column_names[:cols])
        tops = " " * 5 + (" " * 3).join("-" * cols)
        roof = " " * 4 + "/ \\" + "_/ \\" * (cols - 1)
        print(headings), print(tops), print(roof)

        for r in range(rows):
            row_mid = " " * indent
            row_mid += " {} | ".format(r + 1)
            if preds is None:
                row_mid += " | ".join(color_mapping(cell) for cell in board[r])
            else:
                row_mid += " | ".join(
                    color_mapping(board[r][c], r * cols + c) for c in range(cols)
                )
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

    def transform_state_for_nn(self):
        return np.where(self.board == 1, 1, np.where(self.board == 2, -1, 0))

    def get_nn_input(self, isForReplayBuffer=False):
        flat_board = self.transform_state_for_nn().flatten()
        format_player = 1 if self.player_turn == 1 else -1
        nn_input = np.append(flat_board, format_player)
        nn_input = np.array(nn_input, dtype=np.float64)
        if isForReplayBuffer:
            return nn_input
        return np.expand_dims(nn_input, axis=0)

    # make a clone of the board
    def clone(self):
        clone = copy.deepcopy(self)
        return clone

    def get_nn_target(self, nodes: List) -> np.ndarray:
        """Get the target distribution of visit counts for the neural network.

        Args:
            nodes (List['Node']): The nodes from the MCTS.

        Returns:
            np.ndarray: The target distribution of visit counts.
        """
        move_visits = []
        for node in nodes:
            move_visits.append((node.move_from_parent, node.visits))

        visit_counts = [0] * self.board_size**2
        for move, visits in move_visits:
            index = move[0] * self.board_size + move[1]
            visit_counts[index] = visits
        total_visit_count = max(1, sum(visit_counts))
        distribution = np.array([count / total_visit_count for count in visit_counts])
        return distribution

    def transform_nn_target_to_moves(
        self, nn_output: np.ndarray
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Transform the output of the neural network into a list of moves and their visit counts.

        Args:
            nn_output (np.ndarray): The output of the neural network.

        Returns:
            List[Tuple[Tuple[int, int], int]]: A list of moves and their visit counts.
        """
        move_visits = []
        for i, probibility in enumerate(nn_output):
            row = i // self.board_size
            col = i % self.board_size
            move = (row, col)
            move_visits.append((move, probibility))
        return move_visits

    # nn_output is a 2d array of size 1x49
    def get_move_from_nn_output(self, nn_output: np.ndarray) -> Tuple[int, int]:
        """Get the move with the highest visit count from the output of the neural network.

        Args:
            nn_output (np.ndarray): The output of the neural network.

        Returns:
            Tuple[int, int]: The move with the highest visit count.
        """

        # get max valid move
        best_move = None
        max_prob = 0
        for i, prob in enumerate(nn_output[0]):
            row = i // self.board_size
            col = i % self.board_size
            if self.isAvailable_move(row, col) and prob > max_prob:
                max_prob = prob
                best_move = (row, col)
        return best_move

    def get_nn_player(self):
        return 1 if self.player_turn == 1 else -1

    def get_index_from_move(self, move: Tuple[int, int]) -> int:
        """Get the index of a move on the board.

        Args:
            move (Tuple[int, int]): The move to get the index of.

        Returns:
            int: The index of the move on the board.
        """
        return move[0] * self.board_size + move[1]

    def go_to_end_game(self):
        for i in range(self.board_size - 1):
            for j in range(self.board_size):
                self.make_move((i, j))
        if self.board_size == 7:
            self.make_move((6, 6))
            self.make_move((6, 1))

    def get_move_from_preds(self, preds):
        # find index of max value
        # filter out invalid moves
        sorted_preds = np.argsort(preds)
        max_index = None
        for i in range(len(sorted_preds) - 1, -1, -1):
            row = sorted_preds[i] // self.board_size
            col = sorted_preds[i] % self.board_size
            if self.isAvailable_move(row, col):
                max_index = sorted_preds[i]
                break
        row = max_index // self.board_size
        col = max_index % self.board_size
        return (row, col)


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
        # make preds
        preds = np.random.rand(49)
        game.draw_state(preds)
        print(game.check_win())
        print(game.get_player_turn())
        # print(game.get_nn_input())
        # Fore color
        print(f"{Fore.RED}Hello World {Fore.RESET}Hello World")
        # print(game.get_nn_input_advanced())
