"""
This module contains a class to build a neural network model by using tf.Keras
"""

from datetime import datetime
from enum import Enum
from math import sqrt
import os
from typing import List, Tuple

import numpy as np
from config.params import (
    ACTIVATION,
    EPOCHS,
    INPUT_SHAPE,
    LAYERS,
    LEARNING_RATE,
    OPTIMIZER,
    OUTPUT_SHAPE,
)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD, Adagrad
import keras
from game.hex import Hex
from neural_net.enums import Activation, Optimizer
from keras import activations
import shutil

from utils import get_model_location, get_tournament_name


class ANet:
    def __init__(
        self,
        activation: Activation = ACTIVATION,
        optimizer: Optimizer = OPTIMIZER,
        layers: List[int] = LAYERS,
        learning_rate: float = LEARNING_RATE,
        input_shape: int = INPUT_SHAPE,
        output_shape: int = OUTPUT_SHAPE,
        # model = None
    ):
        self.activation = activation
        self.optimizer = optimizer
        self.layers = layers
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model: keras.models.Model = self.build_model()

    def build_model(self) -> keras.models.Model:
        model: keras.models.Model = Sequential()
        # Adding the input layer //TODO: maybe make this with the
        model.add(keras.layers.InputLayer(shape=(self.input_shape,)))

        # Adding hidden layers
        for layer_size in self.layers:
            model.add(Dense(layer_size, activation=self.activation.value))

        # Adding the output layer
        model.add(
            Dense(self.output_shape, activation="softmax")
        )  # Using softmax for multi-class classification

        optimizer = self.get_optimizer()
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(self, x_train, y_train, epochs=EPOCHS):
        self.model.fit(x_train, y_train, epochs=epochs)

    def train_batch(self, batch: List[Tuple]):
        feature_matrix = []
        probability_distribution = []
        for x, D in batch:
            feature_matrix.append(x)
            probability_distribution.append(D)
        feature_matrix = np.array(feature_matrix)
        probability_distribution = np.array(probability_distribution)
        self.train(feature_matrix, probability_distribution)

    def predict(self, x: np.ndarray):
        return self.model.predict(x, verbose=0, batch_size=5000)
        # TODO: check this
        # print("prediction")
        # return self.model.predict(x)

    def save_model(self, train_session, game_name="hex"):
        board_size = int(sqrt(self.output_shape))
        location, params_location = get_model_location(
            board_size, train_session, game_name
        )
        print(location)
        keras.saving.save_model(self.model, location)
        # Copy params file only if it doesn't exist in the target directory
        params_file_location = os.path.join(
            os.path.dirname(__file__), "../config/params.py"
        )
        if not os.path.exists(params_location):
            shutil.copy2(params_file_location, params_location)

    def get_optimizer(self):
        if self.optimizer == Optimizer.ADAGRAD:
            return Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == Optimizer.ADAM:
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer == Optimizer.RMSPROP:
            return RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer == Optimizer.SGD:
            return SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")


def load_model(date, num, game_name="hex", board_size=7):
    path = f"../models/{game_name}/{board_size}x{board_size}/{date}/model_{num}.h5"
    path = os.path.join(os.path.dirname(__file__), path)
    return keras.models.load_model(path)


if __name__ == "__main__":

    def main():
        game = Hex(board_size=7)
        target = np.zeros(game.board_size**2)
        game.go_to_end_game()
        game.draw_state()

        board_rep = game.get_nn_input(True)

        get_legal_moves = game.get_legal_moves()
        for move in get_legal_moves:
            index = game.get_index_from_move(move)
            if move[0] == 6 and move[1] == 0:
                target[index] = 0.80
            else:
                target[index] = 0.05

        print(board_rep)
        print(target)
        anet = ANet(
            input_shape=game.board_size**2 + 1,
            output_shape=game.board_size**2,
        )

        minibatch = [(board_rep, target), (board_rep, target)]

        anet.train_batch(minibatch)
        tournament_name = get_tournament_name(game.board_size)
        anet.save_model(tournament_name)
        # res = anet.predict(np.expand_dims(board_rep, axis=0))
        # next_move = game.get_move_from_nn_output(res)
        # round off to 2 decimal places
        # print(np.round(res, 3))
        # print(next_move)
        # print(game.move_to_str(next_move))
        # assert game.move_to_str(next_move) == "A7"

        # loadedAnet = load_model("2024-04-23", 0, "hex", 7)
        # res = loadedAnet.predict(np.expand_dims(board_rep, axis=0))
        # next_move = game.get_move_from_nn_output(res)
        # assert game.move_to_str(next_move) == "A7"
        # print("All tests passed")

    main()
