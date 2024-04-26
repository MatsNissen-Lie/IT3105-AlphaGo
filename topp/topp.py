import numpy as np
from keras.models import load_model
import os
import tabulate as tb
from game.hex import Hex
from neural_net.onix import ANet2
from utils import get_model_location


class Topp:
    def __init__(self, num_games):
        current_dir = os.getcwd()
        folder_path = current_dir + "/models"
        self.model_path = folder_path  # Replace "/path/to/models/folder" with the actual path to your models folder
        self.num_games = num_games
        # self.model_names = []
        self.models = []
        # I want a list of scores for each model
        self.scores_lists = []

    def init_scores(self):
        # col 1 score col 2 games_started
        self.scores_lists = [[0, 0, 0] for _ in range(len(self.models))]

    def load_models(self, models):
        for model_name in models:
            model = load_model(self.model_path + model_name)
            self.models.append(model)

    def load_models2(self, board_size, folder, max=None):
        next_location = get_model_location(board_size, folder)[0]
        num = int(next_location.split("/")[-1].split("_")[-1].split(".")[0])

        while num != 0:
            num = num - 1
            next_location = next_location.replace(f"model_{num + 1}", f"model_{num}")
            # if max is not None and len(self.models) == max:
            #     break
            # # add only 4 and 0
            # if num % 4 != 0:
            #     continue
            onix = ANet2(model=load_model(next_location))
            self.models.append((onix, next_location.split("/")[-1][:-3]))
            # self.model_names.append(next_location.split("/")[-1][:-3])
        self.init_scores()

    def play_game(self, model1, model2, verbose=False):
        model1, name1 = model1
        model2, name2 = model2
        if verbose:
            print(f"Playing game between {name1} and {name2}")
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
            if verbose:
                game.draw_state()
        winner = game.check_win()
        if winner == 1:
            return 1
        else:
            return -1

    def play_tournament(self):
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                for game_index in range(self.num_games):
                    if game_index % 2 == 0:
                        starter, opponent = i, j
                    else:
                        starter, opponent = j, i

                    # Play the game with the current starter
                    result = self.play_game(self.models[starter], self.models[opponent])

                    # Update game count for both models
                    self.scores_lists[i][2] += 1
                    self.scores_lists[j][2] += 1

                    # Update the total starts count
                    self.scores_lists[starter][1] += 1

                    # Update win count based on the result
                    if result == 1:
                        self.scores_lists[starter][0] += 1  # Starter wins
                    elif result == -1:
                        self.scores_lists[opponent][0] += 1  # Opponent wins

        self.show_results()

    def show_results(self):
        print("\nResults of the tournament")
        model_names = [model[1] for model in self.models]
        results_to_print = [
            [name] + stats for stats, name in zip(self.scores_lists, model_names)
        ]
        print(
            tb.tabulate(
                results_to_print,
                headers=[
                    "Model",
                    "Wins",
                    "Games Started",
                    "Total Games Played",
                ],
                tablefmt="pretty",
            )
        )


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
    tourney.load_models2(4, "train_session0")
    tourney.play_tournament()

    tourney.play_game(tourney.models[0], tourney.models[-1], verbose=True)
    tourney.play_game(tourney.models[-1], tourney.models[0], verbose=True)
