import numpy as np
from keras.models import load_model
import os

from game.hex import Hex
from utils import get_model_location


class Topp:
    def __init__(self, num_games):
        current_dir = os.getcwd()
        folder_path = current_dir + "/models"
        self.model_path = folder_path  # Replace "/path/to/models/folder" with the actual path to your models folder
        self.num_games = num_games
        self.models = []
        # I want a list of scores for each model
        self.scores_lists = []

    def init_scores(self):
        # col 1 score col 2 games_started
        self.scores_lists = [(0, 0) for _ in range(len(self.models))]

    def load_models(self, models):
        for model_name in models:
            model = load_model(self.model_path + model_name)
            self.models.append(model)

    def load_models2(self, board_size, folder):
        next_location = get_model_location(board_size, folder)[0]
        num = int(next_location.split("/")[-1].split("_")[-1].split(".")[0])

        print(f"Loading models from {num}")
        while num != 0:
            num = num - 1
            next_location = next_location.replace(f"model_{num + 1}", f"model_{num}")
            print(f"Loading model from\n {next_location}\n")
            self.models.append(load_model(next_location))
        self.init_scores()

    def play_game(self, model1, model2):

        game = Hex(4)
        while not game.is_terminal():
            if game.player_turn == 1:
                state = game.get_nn_input()
                pred = model1.predict(state)
                move = game.get_move_from_nn_output(pred)
            else:
                state = game.get_nn_input()
                pred = model2.predict(state)
                move = game.get_move_from_nn_output(pred)
            game.make_move(move)
        winner = game.check_win()
        if winner == 1:
            return 1
        else:
            return -1

    def play_tournament(self):
        game_number = 0
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                for _ in range(self.num_games):
                    # result = self.play_game(self.models[i], self.models[j])
                    game_number += 1
                    if game_number % 2 == 0:
                        result = self.play_game(self.models[i], self.models[j])
                    else:
                        result = self.play_game(self.models[j], self.models[i])

                    if result == 1:
                        # Increment score of the model that won the game
                        # If result is 1 and game_number is even, models[i] wins, otherwise models[j] wins
                        winner_index = i if game_number % 2 == 0 else j
                        self.scores_lists[winner_index][0] += 1
                    elif result == -1:
                        loser_index = j if game_number % 2 == 0 else i
                        self.scores_lists[loser_index][0] += 1

        # Print the scores
        for i in range(len(self.models)):
            print(f"Model {i} score: {self.scores[i]}")


if __name__ == "__main__":
    tourney = Topp(4)
    # tourney.load_models(
    #     [
    #         "/hex/4x4/2024-04-24/model_0.h5",
    #         "/hex/4x4/2024-04-24/model_1.h5",
    #         "/hex/4x4/2024-04-24/model_2.h5",
    #         "/hex/4x4/2024-04-24/model_3.h5",
    #         "/hex/4x4/2024-04-24/model_4.h5",
    #         "/hex/4x4/2024-04-24/model_5.h5",
    #     ]
    # )
    tourney.load_models2(4, "tournament3")
    tourney.play_tournament()
