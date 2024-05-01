import copy
from typing import List, Tuple
import numpy as np
from colorama import Back, Style, Fore

# from tree_search.node import Node


class Hex:
    def __init__(self, board_size=7, starting_player=1, rotate_palyer2_for_nn=False):

        if not 3 <= board_size <= 10:
            raise ValueError("Board size must be between 3 and 10")
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.player_turn = starting_player
        self.action = None

        # roateting the board for player 2 makes player 2 and player 1 have the same perspective and objective. Perhaps this makes learning easier for the neural network.
        self.rotate_palyer2_for_nn = rotate_palyer2_for_nn

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
        if preds is not None and self.rotate_palyer2_for_nn and self.player_turn == 2:
            preds = self.rotate_for_nn(preds)

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

    def get_move_from_str(self, move_str: str):
        if move_str.lower() == "pass" or move_str.lower() == "":
            return None
        col = ord(move_str[0].upper()) - 65
        row = int(move_str[1:]) - 1
        if not self.is_valid_move(row, col):
            return None
        return row, col

    def transform_state_for_nn(self, board=None):
        if board is None:
            board = self.board
        return np.where(board == 1, 1, np.where(board == 2, -1, 0)).flatten()

    def get_nn_input(self, isForReplayBuffer=False):
        new_board = (
            self.rotate_board(self.board)
            if self.player_turn == 2 and self.rotate_palyer2_for_nn
            else self.board
        )
        flat_board = self.transform_state_for_nn(new_board)
        format_player = 1 if self.player_turn == 1 else -1
        nn_input = np.append(flat_board, format_player)
        nn_input = np.array(nn_input, dtype=np.float64)
        if isForReplayBuffer:
            return nn_input
        return np.expand_dims(nn_input, axis=0)

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

        if self.player_turn == 2 and self.rotate_palyer2_for_nn:
            distribution = self.rotate_for_nn(distribution)
        return distribution

    def get_move_from_nn_output(self, nn_output: np.ndarray) -> Tuple[int, int]:
        """
        Get the move with the highest visit count from the output of the neural network.

        Args:
            nn_output (np.ndarray): The output of the neural network.

        Returns:
            Tuple[int, int]: The move with the highest visit count.
        """
        nn_output = nn_output[0]
        # make a 2d array like the board of the nn_output
        if self.player_turn == 2 and self.rotate_palyer2_for_nn:
            nn_output = self.rotate_for_nn(nn_output)
        # get max valid move
        best_move = None
        max_prob = 0
        for i, prob in enumerate(nn_output):
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

    def reset(self, starting_player=1):
        self.board = np.zeros((self.board_size, self.board_size))
        self.player_turn = starting_player
        self.action = None

    def rotate_board(self, board):
        old_borad = np.asarray(board)
        new_board = np.zeros((self.board_size, self.board_size))
        for row in range(self.board_size):
            for col in range(self.board_size):
                new_board[col][row] = old_borad[row][col]
        return new_board

    def rotate_for_nn(self, nn_output):
        nn_output = nn_output.reshape(self.board_size, self.board_size)
        nn_output = self.rotate_board(nn_output)
        return nn_output.flatten()


# Uncomment to test the game setup
if __name__ == "__main__":
    game = Hex(7, rotate_palyer2_for_nn=True)
    if False:
        True
    else:
        game.go_to_end_game()
        np.random.seed(2)
        preds = np.random.rand(49)
        preds2 = game.rotate_for_nn(preds)

        # print preds as 7x7 matrix
        print(preds.reshape(7, 7))

        game.draw_state(preds=preds)
        move = game.get_move_from_nn_output([preds])
        game.make_move(move)

        # revert move
        assert game.player_turn == 2
        game.board[move[0]][move[1]] = 0
        game.draw_state()
        move2 = game.get_move_from_nn_output([preds2])

        # denne burde ikke være lik?
        assert move == move2

        print("######################")

        # game.make_move((6, 4))
        # game.board = game.rotate_board()
        # game.draw_state(preds=preds2)
        # game.draw_state()
        # make preds

        # game.board = game.rotate_board()

        # move = game.get_move_from_nn_output([preds])
        # game.make_move(move)
        # game.draw_state()
