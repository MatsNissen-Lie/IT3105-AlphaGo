"""
This is an example of how to use the neural network actors
"""

import argparse

import numpy as np
from game import Hex


def main(args):
    pass


def parse_args():
    """
    Parse command line arguments

    Returns
    -------
    argparse.Namespace
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="An example CLI for neural network actors"
    )

    parser.add_argument(
        "--load_models",
        action="store_true",
        help="Load pre-trained neural network models",
    )

    parser.add_argument("--tournament", action="store_true", help="Run a tournament")

    parser.add_argument(
        "--train", action="store_true", help="Train the neural network model"
    )

    parser.add_argument(
        "--play", action="store_true", help="Play against the neural network model"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
