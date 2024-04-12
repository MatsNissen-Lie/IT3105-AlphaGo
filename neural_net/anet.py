"""
This module contains a class to build a neural network model by using tf.Keras
"""

from enum import Enum


class Optimizer(Enum):
    """
    Optimizer enum
    """

    ADAGRAD = "adagrad"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    SGD = "sgd"


class Activation(Enum):
    """
    Activation enum
    """

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
