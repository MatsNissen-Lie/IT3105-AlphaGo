import random
import time
import pickle
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
    UNIFORM_PALYER,
)

from game.hex import Hex
from neural_net.anet import ANet, load_model
from neural_net.onix import ONIX
from tree_search import MCTS
from utils import get_buffer_location, get_train_session_name, load_kreas_model


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
        self.verbose_sample(batch_size)
        return random.sample(self.buffer, batch_size)

    def verbose_sample(self, batch_size):
        print("Buffer size: ", len(self.buffer))
        print("Sample size: ", batch_size)

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def __len__(self):
        return len(self.buffer)


class Actor:
    def __init__(
        self,
        anet: ONIX = ONIX(),
        replay_buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE),
        board_size: int = BOARD_SIZE,
        game=Hex(BOARD_SIZE),
        simulations: int = SIMULATIONS,
        number_of_games: int = NUMBER_OF_GAMES,
        save_interval: int = SAVE_INTERVAL,
        epsilon_decay: float = EPSILON_DECAY,
        min_epsilon: float = MIN_EPSILON,
        startEpsilon: bool = False,
    ) -> None:
        self.anet = anet
        self.replay_buffer = replay_buffer
        self.game = game
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

    def train(self, session_name: str = "session"):
        session_time = time.time()
        train_session = get_train_session_name(self.board_size, session_name)

        print(f"Number of games: {self.number_of_games}")
        for game_number in range(self.number_of_games):

            starting_player = 1 if game_number % 2 == 0 else 2
            self.game.reset(starting_player)
            game = self.game

            epsilon = self.epsiolon_decay(game_number)
            mcts = MCTS(game, self.anet, self.simulations)
            root = mcts.get_root()

            print(f"\n\nGame {game_number+1} started. Epsilon: {epsilon}")

            start_time = time.time()
            while not game.is_terminal():

                # mcts run sets a new root node and discards everything else in the tree
                best_node, child_nodes = mcts.run(root, epsilon)
                X, Y = game.get_nn_input(True), game.get_nn_target(child_nodes)
                self.replay_buffer.add(X, Y)

                game.draw_state(Y)
                game.make_move(best_node.move_from_parent)
                root = best_node
                print("\n")

            time_taken = time.time() - start_time
            game.draw_state()
            print(f"Game {game_number+1} finished. Winner: {game.check_win()}")
            print(f"Starting player: {starting_player}")
            print(
                f"Time taken: {time_taken//60:.0f}m {time_taken%60:.0f}s and {np.round((time_taken%60*1000%1000))}ms"
            )
            # print(f"Total time taken: {time_taken*1000:.2f}ms")

            minibatch = self.replay_buffer.sample()
            self.anet.train_batch(minibatch)

            if (game_number + 1) % self.save_interval == 0:
                self.anet.save_model(train_session=train_session)

        buffer_location = get_buffer_location(self.board_size, train_session)
        self.replay_buffer.save_to_file(buffer_location)

        session_time = time.time() - session_time
        print(f"Total time taken: {session_time//3600:.0f}h {session_time//60:.0f}m")

    def play(self, myPlayer: int = 1):
        game = Hex(self.board_size)
        while not game.is_terminal():
            if game.get_player() == myPlayer:
                game.draw_state()
                move = game.get_move_from_str(input("Enter move: "))
            else:
                pred = self.anet.predict(game.get_nn_input())
                move = game.get_move_from_nn_output(pred)
            game.make_move(move)
        game.draw_state()
        print(f"Winner: {game.check_win()}")


if __name__ == "__main__":
    train = True
    train = False

    train_continue = True
    test_replaybuffer = False
    test_simulation_time = True

    if train:
        # continue traning
        game = Hex(board_size=BOARD_SIZE, rotate_palyer2_for_nn=UNIFORM_PALYER)
        anet = ONIX(
            input_shape=game.get_input_shape(), output_shape=game.get_output_shape()
        )
        actor = Actor(anet=anet, game=game)
        actor.train("kungfu")

    elif train_continue:
        print("Continue training")
        model = load_kreas_model("train_session3", 18, "hex", 7)
        anet = ONIX(model)
        game = Hex(board_size=BOARD_SIZE, rotate_palyer2_for_nn=UNIFORM_PALYER)
        actor = Actor(
            anet=anet, game=game, epsilon_decay=0.1, min_epsilon=0.1, startEpsilon=True
        )
        actor.train("train_session4_extended")

    elif test_simulation_time:
        game = Hex(7)
        anet = ONIX(input_shape=game.board_size**2 + 1, output_shape=game.board_size**2)
        actor = Actor(
            anet=anet,
            simulations=3000,
            board_size=game.board_size,
            number_of_games=1,
            save_interval=10,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            startEpsilon=True,
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
                "random",
                "7x7",
                "124x124x124",
                "onnix",
                1030,
            ],
            [
                "ANet2",
                "7x7",
                "124x124x124",
                "onnix_evolved",
                84,
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
