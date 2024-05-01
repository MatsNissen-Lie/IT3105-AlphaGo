"""
General configuration
"""

from config.enums import Activation, Optimizer

"""
Configuration of board
"""
BOARD_SIZE = 7
UNIFORM_PALYER = False

"""
Configuration of neural network model
"""
INPUT_SHAPE = BOARD_SIZE * BOARD_SIZE + 1
OUTPUT_SHAPE = BOARD_SIZE * BOARD_SIZE
LAYERS = [512, 256, 256]
ACTIVATION = Activation.RELU
OPTIMIZER = Optimizer.ADAM
LEARNING_RATE = 1e-3
EPOCHS = 1
"""
configuration of the MCTS
"""
TIME_LIMIT = 10
EXPLORATION_CONSTANT = 1

"""
This file contains the configuration for the reinforcement learning algorithm.
"""
REPLAY_BUFFER_SIZE = 2048
REPLAY_BATCH_SIZE = 256
NUMBER_OF_GAMES = 50
# SIMULATIONS = 100
SIMULATIONS = 1500
# IDENTIFIER = "model"
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.1
SAVE_INTERVAL = 10  # NUMBER_OF_GAMES // 5
