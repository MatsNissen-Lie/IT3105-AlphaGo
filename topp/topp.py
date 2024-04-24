import numpy as np
from keras.models import load_model
import os

from game.hex import Hex

class Topp:
    def __init__(self, num_games):
        current_dir = os.getcwd()  
        folder_path = current_dir+"/models" 
        self.model_path = folder_path # Replace "/path/to/models/folder" with the actual path to your models folder
        self.num_games = num_games
        self.models = []
        self.scores = np.zeros(len(self.model_path))

    def load_models(self, models):
        for model_name in models:
            model = load_model(self.model_path + model_name)
            self.models.append(model)

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
        else: return -1

    def play_tournament(self):
        for i in range(len(self.models)):
            for j in range(i+1, len(self.models)):
                for _ in range(self.num_games):
                    result = self.play_game(self.models[i], self.models[j])
                    if result == 1:
                        self.scores[i] += 1
                    elif result == -1:
                        self.scores[j] += 1

        # Print the scores
        for i in range(len(self.models)):
            print(f"Model {i} score: {self.scores[i]}")
            
    
    
tourney = Topp(4)
tourney.load_models(["/hex/4x4/2024-04-24/model_0.h5", "/hex/4x4/2024-04-24/model_1.h5", "/hex/4x4/2024-04-24/model_2.h5", "/hex/4x4/2024-04-24/model_3.h5", "/hex/4x4/2024-04-24/model_4.h5", "/hex/4x4/2024-04-24/model_5.h5"])
tourney.play_tournament()
            
            

