import argparse
from neural_net.actor import Actor

from neural_net.onix import ONIX
from topp.topp import Topp
from utils import get_model_location, load_kreas_model


def main(args):
    """
    Main function for the CLI

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    """

    if args.load_models:
        # Load the pre-trained models
        pass

    if args.tournament:
        tourney = Topp(4, switch_starting_player=True, verbose=False)
        tourney.load_models(7, "train_session2", max=10, identifier="boobs")
        tourney.load_models(7, "train_session3", max=10, identifier="killah")
        tourney.load_models(7, "mmv", max=10, identifier="mmv")
        tourney.play_tournament()

    if args.train:
        model = ONIX()
        actor = Actor(model)
        actor.train()

    if args.play:
        kreas_model = load_kreas_model("train_session3", 18, "hex", 7)
        model = ONIX(kreas_model)
        actor = Actor(model)
        actor.play()


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
