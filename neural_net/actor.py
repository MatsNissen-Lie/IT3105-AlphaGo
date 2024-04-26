import random
import time

import numpy as np

from config.params import (
    BOARD_SIZE,
    EPSILON_DECAY,
    MIN_EPSILON,
    NUMBER_OF_GAMES,
    REPLAY_BUFFER_SIZE,
    SAVE_INTERVAL,
    SIMULATIONS,
    REPLAY_BATCH_SIZE,
)

from game.hex import Hex
from anet import ANet, load_model
from onix import ANet2
from tree_search import MCTS
from utils import get_train_session_name


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int = REPLAY_BUFFER_SIZE,
        replay_batch_size: int = REPLAY_BATCH_SIZE,
    ) -> None:
        self.buffer_size = buffer_size
        self.replay_batch_size = replay_batch_size
        self.buffer = []

    def add(self, state, target):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, target))

    def sample(self):
        batch_size = min(len(self.buffer), self.replay_batch_size)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Actor:
    def __init__(
        self,
        anet: any,
        replay_buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE),
        board_size: int = BOARD_SIZE,
        simulations: int = SIMULATIONS,
        number_of_games: int = NUMBER_OF_GAMES,
        save_interval: int = SAVE_INTERVAL,
        epsilon_decay: float = EPSILON_DECAY,
        min_epsilon: float = MIN_EPSILON,
        startEpsilon: bool = False,
    ) -> None:
        self.anet = anet
        self.replay_buffer = replay_buffer
        self.simulations = simulations
        self.number_of_games = number_of_games
        self.save_interval = save_interval
        self.startEpsilon = startEpsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.board_size = board_size

    def epsiolon_decay(self, game_count: int):
        return max(
            self.epsilon_decay
            ** (game_count if not self.startEpsilon else game_count + 1),
            self.min_epsilon,
        )

    def train(self):
        # how many games
        print(f"Number of games: {self.number_of_games}")
        train_session = get_train_session_name(self.board_size)
        for game_number in range(self.number_of_games):
            # for the first iteration epsoilon is 1. No neural network is used. After the first iteration, the epsilon is decayed.
            epsilon = self.epsiolon_decay(game_number)
            starting_player = 1 if game_number % 2 == 0 else 2
            game = Hex(self.board_size, starting_player)
            mcts = MCTS(game, self.anet, self.simulations)
            root = mcts.get_root()
            print(f"Game {game_number+1} started. Epsilon: {epsilon}")
            # take the time of each game

            start_time = time.time()
            while not game.is_terminal():
                # mcts run sets a new root node and discards everything else in the tree
                best_node, child_nodes = mcts.run(root, epsilon)
                X, Y = game.get_nn_input(True), game.get_nn_target(child_nodes)
                self.replay_buffer.add(X, Y)
                # print move from parent
                # print(f"\nPlayer {game.get_player()}: {best_node.move_from_parent}")
                game.draw_state(Y)
                game.make_move(best_node.move_from_parent)
                root = best_node
                # print move probilities
                # mcts.draw_tree()
            time_taken = time.time() - start_time
            game.draw_state()
            print(f"Game {game_number+1} finished. Winner: {game.check_win()}")
            print(f"Starting player: {starting_player}")
            print(
                f"Time taken: {time_taken//60:.0f}m {time_taken%60:.0f}s and {(time_taken%60*1000%1000):.2f}ms"
            )
            # print(f"Total time taken: {time_taken*1000:.2f}ms")

            minibatch = self.replay_buffer.sample()
            self.anet.train_batch(minibatch)

            if (game_number + 1) % self.save_interval == 0:
                self.anet.save_model(train_session=train_session)


if __name__ == "__main__":
    train = True
    train = False
    test_replaybuffer = False
    test_simulation_time = True

    if train:
        anet = ANet2()
        actor = Actor(anet=anet)
        actor.train()
    elif test_simulation_time:
        game = Hex(7)
        anet = ANet2(
            input_shape=game.board_size**2 + 1, output_shape=game.board_size**2
        )
        actor = Actor(
            anet=anet,
            simulations=1,
            board_size=game.board_size,
            number_of_games=1,
            save_interval=10,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            startEpsilon=True,
        )
        actor.train()
    elif not test_replaybuffer:
        anet = load_model("2024-04-23", 0, "hex", 7)
        actor = Actor(
            anet, number_of_games=1, save_interval=1, deleayEpsilon=True, simulations=10
        )
        actor.train()

    else:
        anet = ANet()
        game = Hex()
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

        buffer = ReplayBuffer()
        buffer.add(board_rep, target)
        buffer.add(board_rep, target)
        buffer.sample()
        anet.train_batch(buffer.sample())

        res = anet.predict(np.expand_dims(board_rep, axis=0))
        next_move = game.get_move_from_nn_output(res)
        # round off to 2 decimal places
        print(np.round(res, 3))
        print(next_move)
        print(game.move_to_str(next_move))
        assert game.move_to_str(next_move) == "A7"

    def print_reqords():
        from config.params import BOARD_SIZE, LAYERS

        print(f"Board size: {BOARD_SIZE}")
        print(f"Layers: {LAYERS}")
        array_time = [
            # model name, borad size, nural net size, anet name, ms per rollout
            [
                "model_4",
                "4x4",
                "124x124x124",
                "onnix",
                1030,
            ],
            [
                "model_4",
                "4x4",
                "124x124x124",
                "onnix",
                1130,
            ],
        ]
        import tabulate

        print(
            tabulate.tabulate(
                array_time,
                headers=[
                    "Model Name",
                    "Board Size",
                    "Neural Net Size",
                    "ANet Name",
                    "ms per sim",
                ],
                tablefmt="pretty",
            )
        )
