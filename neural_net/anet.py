"""
This module contains a class to build a neural network model by using tf.Keras
"""

import datetime
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

activations.relu


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
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

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
        feature_matrix = np.array([])
        probability_distribution = np.array([])
        for x, D in batch:
            feature_matrix.append(x)
            probability_distribution.append(D)
        # WHY DO we train on an array?
        self.train(feature_matrix, probability_distribution)

    def predict(self, x: np.ndarray):
        return self.model.predict(x)

    def save_model(self):
        num = 0
        board_size = sqrt(self.output_shape)
        date = datetime.now().strftime("%Y-%m-%d")
        location = f"models/{board_size}x{board_size}/{date}/model_{num}.h5"
        while os.path.exists(location):
            num += 1
            location = f"models/model_{num}.h5"
        self.model.save(location)

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


if __name__ == "__main__":

    def main():
        game = Hex()
        target = np.zeros(game.board_size**2)
        target = game.get_nn_input()
        game.go_to_end_game()
        # target = np.append(target, game.get_nn_player())
        target[-1] = game.get_nn_player()
        game.draw_state()

        board_rep = game.get_nn_input()

        get_legal_moves = game.get_legal_moves()
        for move in get_legal_moves:
            index = game.get_index_from_move(move)
            if move[0] == 6 and move[1] == 0:
                target[index] = 0.80
            else:
                target[index] = 0.05
        # append the player

        print(target)
        # anet = ANet()

    main()
