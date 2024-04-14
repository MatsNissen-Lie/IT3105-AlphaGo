import random
from config.params import (
    BOARD_SIZE,
    EPSILON_DECAY,
    NUMBER_OF_GAMES,
    REPLAY_BUFFER_SIZE,
    SIMULATIONS,
)
from game.hex import Hex
from neural_net.anet import ANet
from tree_search import MCTS


class ReplayBuffer:
    def __init__(self, buffer_size: int = 2048) -> None:
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 256):
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
    ) -> None:
        self.anet = anet
        self.replay_buffer = replay_buffer
        self.simulations = simulations
        self.number_of_games = number_of_games

    def epsiolon_decay(self, game_count: int):
        return EPSILON_DECAY ** (game_count)

    def play(self):
        for _ in range(self.number_of_games):
            # for the first iteration epsoilon is 1. No neural network is used. After the first iteration, the epsilon is decayed.
            epsilon = self.epsiolon_decay(_)
            game = Hex(BOARD_SIZE)
            mcts = MCTS(game, self.anet, self.simulations, epsilon)

            while not game.is_terminal():
                move = mcts.run(game)
