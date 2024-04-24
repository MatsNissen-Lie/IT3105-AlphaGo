"""
General configuration
"""

from neural_net.enums import Activation, Optimizer


BOARD_SIZE = 4
TIME_LIMIT = 2
MAX_TIME_LIMIT = 7
NUM_OF_MODELS = 6

"""
Configuration of neural network model
"""
INPUT_SHAPE = BOARD_SIZE * BOARD_SIZE + 1
OUTPUT_SHAPE = BOARD_SIZE * BOARD_SIZE
LAYERS = [99, 99]
ACTIVATION = Activation.RELU
OPTIMIZER = Optimizer.ADAM
LEARNING_RATE = 1e-3
EPOCHS = 100

"""
This file contains the configuration for the reinforcement learning algorithm.
"""
REPLAY_BUFFER_SIZE = 2048
REPLAY_BATCH_SIZE = 256
NUMBER_OF_GAMES = 100
# SIMULATIONS = 100
SIMULATIONS = 1000
# IDENTIFIER = "model"
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.1
SAVE_INTERVAL = 10
