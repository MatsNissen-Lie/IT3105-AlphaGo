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
        self.activateion = activation
        self.optimizer = optimizer
        self.layers = layers
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Adding the input layer //TODO: maybe make this with the
        model.add(keras.layers.InputLayer(input_shape=(self.input_shape,)))

        # Adding hidden layers
        for layer_size in self.layers:
            model.add(Dense(layer_size, activation=self.activation))

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
        self.train(np.array(feature_matrix), np.array(probability_distribution))

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
            return Adagrad(lr=self.learning_rate)
        elif self.optimizer == Optimizer.ADAM:
            return Adam(lr=self.learning_rate)
        elif self.optimizer == Optimizer.RMSPROP:
            return RMSprop(lr=self.learning_rate)
        elif self.optimizer == Optimizer.SGD:
            return SGD(lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")


if __name__ == "__main__":

    def main():
        game = Hex()
        flat_board = game.transform_board_values_for_nn().flatten()
        player_to_move = [1 if game.player_turn == 1 else -1]
        board_representation = np.hstack([flat_board, player_to_move])
        if True:
            return board_representation
        return np.expand_dims(board_representation, axis=0)

    print(main())
