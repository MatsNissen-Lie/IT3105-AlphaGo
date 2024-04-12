"""
This is an example of how to use the neural network actors
"""

import argparse

import numpy as np
from game import Hex
from mcts import Node
from neural_network import load_models
from neural_network.anet import ANet
from reinforcement_learning import Actor
from config import IDENTIFIER, BOARD_SIZE, NUM_OF_MODELS
from topp import TOPP
