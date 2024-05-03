import argparse
from config.params import BOARD_SIZE, UNIFORM_PALYER
from game.hex import Hex
from neural_net.actor import Actor

from neural_net.onix import ONIX
from topp.topp import Topp
from utils import load_kreas_model


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
        # tourney.load_models(7, "mmv", max=5, identifier="mmv")
        tourney.load_models(7, "train_session3", max=5, identifier="explore")
        tourney.load_models(7, "train_session4", max=5, identifier="xp")

        tourney.load_models(
            7, "train_session4_extended1", max=10, identifier="extended"
        )

        # load two models and play them against each other
        # tourney.load_model(7, "rotate0", 4, "50rotate")
        # tourney.load_model(7, "rotate0", 3, "40rotate")
        # tourney.play_game(tourney.models[0], tourney.models[-1], verbose=True)
        # tourney.play_game(tourney.models[-1], tourney.models[0], verbose=True)

        tourney.play_tournament()

    if args.train:
        model = ONIX()
        game = Hex(BOARD_SIZE, rotate_palyer2_for_nn=UNIFORM_PALYER)
        actor = Actor(model, game=game)
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
