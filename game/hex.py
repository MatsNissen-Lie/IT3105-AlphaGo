class Hex:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.player_turn = 1

    def is_valid_move(self, row, col):
        # Check if the move is within the board and the cell is empty
        return (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row][col] == 0
        )

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.player_turn
            self.player_turn = 3 - self.player_turn  # Switch player
            return True
        return False

    # Placeholder for the win check function
    def check_win(self):
        # This method will be implemented to check if the current player has won
        pass

    def play(self):
        # Game loop - to be implemented
        pass

    def print_board(self):
        # Printing the Hex board with proper indentation
        for row in range(self.board_size):
            print(" " * row, end="")  # Indentation for hexagonal appearance
            for col in range(self.board_size):
                piece = self.board[row][col]
                if piece == 0:
                    print(". ", end="")
                elif piece == 1:
                    print("X ", end="")
                else:
                    print("O ", end="")
            print()


# Uncomment to test the game setup
# game = HexGame()
# game.print_board()
# game.make_move(0, 0)
# game.print_board()
# print("Current player:", game.player_turn)
