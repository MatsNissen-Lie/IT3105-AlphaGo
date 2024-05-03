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
    def __init__(self, num_games, switch_starting_player=False, verbose=False):
        current_dir = os.getcwd()
        folder_path = current_dir + "/models"
        self.model_path = folder_path  # Replace "/path/to/models/folder" with the actual path to your models folder
        self.num_games = num_games
        # self.model_names = []
        self.models = []
        # I want a list of scores for each model
        self.scores_lists = []
        self.switch_starting_player = switch_starting_player
        self.verbose = verbose

    def init_scores(self):
        # col 1 score col 2 games_started
        self.scores_lists = [[0, 0, 0, 0] for _ in range(len(self.models))]

    def load_model(self, board_size, folder, num, new_name):
        onix = ONIX(model=load_model_onix(folder, num, "hex", board_size))
        self.models.append((onix, f"{new_name}"))
        self.scores_lists.append([0, 0, 0, 0])

    def load_models(self, board_size, folder, max=None, identifier=""):
        next_location = get_model_location(board_size, folder)[0]
        num = int(next_location.split("/")[-1].split("_")[-1].split(".")[0])
        loaded = 0
        while num != 0:
            if max is not None and loaded == max:
                break
            num = num - 1
            next_location = next_location.replace(f"model_{num + 1}", f"model_{num}")
            onix, loaded = ONIX(model=load_model(next_location)), loaded + 1
            name = next_location.split("/")[-1][:-3]
            # replace model with identifier
            name = name.replace("model", identifier)
            self.models.append((onix, name))
            # self.model_names.append(next_location.split("/")[-1][:-3])
        self.init_scores()

    def play_game(self, palyer1, player2, starting_player=1, verbose=False):

        model1, name1 = palyer1
        model2, name2 = player2
        assert model1.output_shape == model2.output_shape
        game = Hex(
            int(sqrt(model1.output_shape)),
            rotate_palyer2_for_nn=True if "rotate" in name2 else False,
        )

        verbose = self.verbose or verbose
        if verbose:
            print(f"Playing game between {name1} and {name2}")
            starting_color = game.get_palyer_color(starting_player)
            print(f"Starting color: {starting_color}")
            if starting_player == 1:
                print(f"{name1} starts")
            else:
                print(f"{name2} starts")

        game.player_turn = starting_player
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

        if verbose:
            game.draw_state()
            print(f"Winner: {name1 if winner == 1 else name2}")
            print("\n")

        if winner == 1:
            return 1
        else:
            return -1

    def play_tournament(self):
        starting_player = 2 if self.switch_starting_player else 1
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                for game_index in range(self.num_games):
                    if game_index % 2 == 0:
                        starter, opponent = i, j
                    else:
                        starter, opponent = j, i

                    if self.switch_starting_player and game_index % 2 == 0:
                        starting_player = 3 - starting_player

                    # Play the game with the current starter
                    result = self.play_game(
                        self.models[starter],
                        self.models[opponent],
                        starting_player,
                    )

                    # Update game count for both models
                    self.scores_lists[i][3] += 1
                    self.scores_lists[j][3] += 1

                    # Update the total starts count
                    if starting_player == 1:
                        self.scores_lists[starter][1] += 1
                    else:
                        self.scores_lists[starter][2] += 1

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
                    "Games Started as Blue",
                    "Games Started as Red",
                    "Total Games Played",
                ],
                tablefmt="pretty",
            )
        )


if __name__ == "__main__":
    tourney = Topp(4, switch_starting_player=True, verbose=False)
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
    # tourney.load_models(7, "train_session1", identifier="OG", max=3)
    tourney.load_models(7, "train_session2", max=10, identifier="boobs")
    tourney.load_models(7, "train_session3", max=10, identifier="killah")
    tourney.load_models(7, "mmv", max=10, identifier="mmv")

    tourney.play_tournament()

    # konklusjon. er at det ikke har så mye å si man bytter på hvem som starter

    # tourney.load_model(7, "train_session2", 1, "20games")
    # tourney.load_model(7, "train_session1", 3, "model_3")
    # tourney.play_game(tourney.models[0], tourney.models[-1], verbose=True)
    # tourney.play_game(tourney.models[-1], tourney.models[0], verbose=True)
