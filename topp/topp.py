from math import sqrt
import numpy as np
from keras.models import load_model
from neural_net.onix import load_model as load_model_onix
import os
import tabulate as tb
from game.hex import Hex
from neural_net.onix import ONIX
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

    def load_model(self, board_size, folder, num, new_name):
        onix = ONIX(model=load_model_onix(folder, num, "hex", board_size))
        self.models.append((onix, f"{new_name}"))
        self.scores_lists.append([0, 0, 0])

    def load_models(self, board_size, folder, max=None, identifier=""):
        next_location = get_model_location(board_size, folder)[0]
        num = int(next_location.split("/")[-1].split("_")[-1].split(".")[0])

        while num != 0:
            num = num - 1
            next_location = next_location.replace(f"model_{num + 1}", f"model_{num}")
            if max is not None and len(self.models) == max:
                break
            onix = ONIX(model=load_model(next_location))
            name = next_location.split("/")[-1][:-3]
            # replace model with identifier
            name = name.replace("model", identifier)
            self.models.append((onix, name))
            # self.model_names.append(next_location.split("/")[-1][:-3])
        self.init_scores()

    def play_game(self, model1, model2, verbose=False):
        model1, name1 = model1
        model2, name2 = model2
        if verbose:
            print(f"Playing game between {name1} and {name2}")
        assert model1.output_shape == model2.output_shape
        game = Hex(int(sqrt(model1.output_shape)))
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
                game.draw_state(pred[0])
        winner = game.check_win()
        if verbose:
            # model won
            print(f"Winner: {name1 if winner == 1 else name2}")
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
        results_to_print.sort(key=lambda x: x[1], reverse=True)
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
    # tourney.load_models(7, "train_session2_noswitch", identifier="noswitch_")

    # tourney.play_tournament()
    # old_models = tourney.models
    # tourney.models = []
    # tourney.load_models(7, "train_session3_switch", identifier="switch_")
    # tourney.play_tournament()

    # # paly all models against each other
    # tourney.models = old_models + tourney.models
    # tourney.init_scores()

    # # load
    tourney.load_models(7, "train_session1", identifier="heavy")
    tourney.load_models(7, "train_session2", max=10, identifier="boobs")

    tourney.play_tournament()

    # konklusjon. er at det ikke har så mye å si man bytter på hvem som starter

    # tourney.load_model(7, "train_session2", 1, "20games")
    # tourney.load_model(7, "train_session1", 3, "model_3")
    # tourney.play_game(tourney.models[0], tourney.models[-1], verbose=True)
    # tourney.play_game(tourney.models[-1], tourney.models[0], verbose=True)
