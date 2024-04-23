import random

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
from neural_net.anet import ANet, load_model
from tree_search import MCTS


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
        anet: ANet,
        replay_buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE),
        simulations: int = SIMULATIONS,
        number_of_games: int = NUMBER_OF_GAMES,
        save_interval: int = SAVE_INTERVAL,
        deleayEpsilon: bool = False,
    ) -> None:
        self.anet = anet
        self.replay_buffer = replay_buffer
        self.simulations = simulations
        self.number_of_games = number_of_games
        self.save_interval = save_interval
        self.deleayEpsilon = deleayEpsilon

    def epsiolon_decay(self, game_count: int):
        return max(
            EPSILON_DECAY ** (game_count if not self.deleayEpsilon else game_count + 1),
            MIN_EPSILON,
        )

    def train(self):
        for game_number in range(self.number_of_games):
            # for the first iteration epsoilon is 1. No neural network is used. After the first iteration, the epsilon is decayed.
            epsilon = self.epsiolon_decay(game_number)
            starting_player = 1 if game_number % 2 == 0 else 2
            game = Hex(BOARD_SIZE, starting_player)
            mcts = MCTS(game, self.anet, self.simulations)
            root = mcts.get_root()
            while not game.is_terminal():
                # mcts run sets a new root node and discards everything else in the tree
                best_node, child_nodes = mcts.run(root, epsilon)
                X, Y = game.get_nn_input(True), game.get_nn_target(child_nodes)
                self.replay_buffer.add(X, Y)
                # print move from parent
                print(f"\nPlayer {game.get_player()}: {best_node.move_from_parent}")
                game.make_move(best_node.move_from_parent)
                root = best_node
                game.draw_state()
                # mcts.draw_tree()
            print(f"Game {game_number} finished. Winner: {game.check_win()}")

            minibatch = self.replay_buffer.sample()
            self.anet.train_batch(minibatch)

            if (game_number + 1) % self.save_interval == 0:
                self.anet.save_model()


if __name__ == "__main__":
    test_replaybuffer = False
    train = True
    if train:
        anet = ANet()
        actor = Actor(anet=anet)
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
