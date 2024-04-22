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


# Pseudocode for the entire algorithm appears below:
# 1. is = save interval for ANET (the actor network) parameters
# 2. Clear Replay Buffer (RBUF)
# 3. Randomly initialize parameters (weights and biases) of ANET 4. For ga in number actual games:
# (a) Initialize the actual game board (Ba) to an empty board.
# (b) sinit ← starting board state
# (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit (d) While Ba not in a final state:
# • Initialize Monte Carlo game board (Bmc) to same state as root. • For gs in number search games:
# – Use tree policy Pt to search from root to a leaf (L) of MCT. Update Bmc with each move.
# – Use ANET to choose rollout actions from L to a final state (F). Update Bmc with each move. – Perform MCTS backpropagation from F to root.
# • next gs
# • D = distribution of visit counts in MCT along all arcs emanating from root. • Add case (root, D) to RBUF
# • Choose actual move (a*) based on D
# • Perform a* on root to produce successor state s*
# • Update Ba to s*
# • In MCT, retain subtree rooted at s*; discard everything else.
# • root ← s*
# (e) Train ANET on a random minibatch of cases from RBUF (f) if ga modulo is == 0:
# • Save ANET’s current parameters for later use in tournament play. 5. next ga


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

            while not game.is_terminal():
                # mcts run sets a new root node and discards everything else in the tree
                best_node, move_visits = mcts.run(root, epsilon)
                X, Y = game.get_nn_input(), game.transform_nn_output(move_visits)
                self.replay_buffer.add(X, Y)

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
