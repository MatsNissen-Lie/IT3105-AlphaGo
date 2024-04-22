import random
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
from neural_net.anet import ANet
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

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

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
    ) -> None:
        self.anet = anet
        self.replay_buffer = replay_buffer
        self.simulations = simulations
        self.number_of_games = number_of_games
        self.save_interval = save_interval

    def epsiolon_decay(self, game_count: int):
        return max(EPSILON_DECAY ** (game_count), MIN_EPSILON)

    def train(self):
        for game_number in range(self.number_of_games):
            # for the first iteration epsoilon is 1. No neural network is used. After the first iteration, the epsilon is decayed.
            epsilon = self.epsiolon_decay(game_number)
            game = Hex(BOARD_SIZE)
            mcts = MCTS(game, self.anet, self.simulations)
            root = mcts.get_root()
            mcts.draw_tree()
            while not game.is_terminal():
                # mcts run sets a new root node and discards everything else in the tree
                print("yum")
                best_node, child_nodes = mcts.run(root, epsilon)
                print("yum2")
                X, Y = game.get_nn_input(), game.get_nn_target(child_nodes)
                print("yum3")
                self.replay_buffer.add(X, Y)
                print("yum4")

                game.make_move(best_node.move_from_parent)
                root = best_node

                print(f"\nPlayer {game.get_player()}: {best_node.move_from_parent}")
                mcts.draw_tree()
            print(f"Game {game_number} finished. Winner: {game.check_win()}")

            minibatch = self.replay_buffer.sample()
            self.anet.train_batch(minibatch)

            if (game_number + 1) % self.save_interval == 0:
                self.anet.save_model()


if __name__ == "__main__":
    anet = ANet()
    actor = Actor(anet, number_of_games=1)
    actor.train()
