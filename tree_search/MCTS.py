import copy
import random
from game.game_interface import GameInterface
from game.hex import Hex
from game.nim import Nim
from tree_search.policy import DefaultPolicy, TreePlolicy
from tree_search.node import Node
from PrettyPrint import PrettyPrintTree


class MCTS:
    def __init__(self, neural_net, iteration_limit, game: GameInterface, M=500):
        self.root = Node(game)
        self.tree_policy = TreePlolicy()
        self.iteration_limit = iteration_limit
        self.M = M  # Number of rollouts
        self.NN = neural_net
        self.NN_confidence = 0.3  # Starting value
        # self.game = game
        # self.player = game.get_player()
        # self.tree_policy = TreePlolicy(self.NN_confidence)

    def select_node(self) -> "Node":
        node, _ = self.tree_policy.search(self.root)
        return node

    def expand(self, node: Node):
        # Get the possible moves from the game state
        possible_moves = node.game_state.get_legal_moves()
        random.shuffle(possible_moves)
        # TODO: Vi kan bruke Anet til å velge hvilke moves vi skal legge til i treet!
        for move in possible_moves:
            old_state = node.game_state.clone()
            old_state.make_move(move)
            node.add_child(old_state, move)

    # aka leaf_evaluation
    def simulate(self, node):
        # Use the critic to evaluate the node
        # her kan du gjør en rollout med Anet som actor eller en critic med Anet og spare deg for rollout.
        """
        Perform 100 rollouts from the given node and average their results.

        Args:
        node (Node): The node from which the rollouts start.

        Returns:
        float: The average value of the 500 rollouts.
        """
        total_value = 0
        num_rollouts = 100

        for _ in range(num_rollouts):
            value = self.rollout(node)
            total_value += value

        average_value = total_value / num_rollouts
        return average_value

    def rollout(self, node: Node, epsilon=0.1, isRandom=True):
        """
        Perform a rollout from the given node using an epsilon-greedy strategy.

        Args:
        node (Node): The node from which the rollout starts.
        epsilon (float): The probability of taking a random action.
        isRandom (bool): Whether to use a random policy for the rollout.
        Returns:
        float: The estimated value of the node.
        """
        # leaf_node: Node = copy.deepcopy(node)
        leaf_node: Node = node
        simulation = DefaultPolicy()
        terminal_state = simulation(leaf_node)
        winner = terminal_state.check_win()
        # we minimize for player 2 and maximize for player 1
        return winner if winner != 2 else -1

    def backpropagate(self, node: Node, value):
        while node is not None:
            node.update(value)
            node = node.parent

    def run(self, initial_state):
        self.root = Node(initial_state)
        for _ in range(self.iteration_limit):
            # make the loop promt the terminal too continue
            # if _ > 3:
            #     input("Press Enter to continue...")
            #     self.draw_tree()
            node = self.select_node()
            if not node.is_terminal():
                self.expand(node)

            value = self.simulate(node)
            self.backpropagate(node, value)
        # Return the best move based on the search
        return self.best_moves()

    def best_moves(self):
        # Implement logic to choose the best move from the root node
        # get children from root node and order them by visits
        moves = []
        for child in self.root.children:
            moves.append((child.game_state.get_state(), child.visits))
        return moves

    # jeg vil gjøre et approch der vi begynner med å gjøre rollouts også gir vi mer og mer tillit til modellen vår etterhvert som vi har gjort flere rollouts
    def update_critic_confidence(self, critic_accuracy):
        # Adjust the confidence based on the accuracy
        increment = 0.02  # Increment value
        decrement = 0.01  # Decrement value
        threshold = 0.75  # Performance threshold (e.g., 75% accuracy)
        # This is a simplistic adjustment logic; you might want a more sophisticated method
        if critic_accuracy > threshold:  # Define a suitable threshold
            self.NN_confidence = min(self.NN_confidence + increment, 1.0)
        else:
            self.NN_confidence = max(self.NN_confidence - decrement, 0.0)

    def draw_tree(
        self,
        child_count=2,
        object_view=lambda x: (
            x.UTC(),
            x.visits,
            "p" + str(x.game_state.get_player()),
        ),
        depth=3,
    ):
        pt = PrettyPrintTree(
            lambda x: x.samle_children(child_count), object_view, max_depth=depth
        )
        pt(self.root)


if __name__ == "__main__":
    game = Hex()
    game.go_to_end_game()
    mcts = MCTS(None, 2000, game)
    res = mcts.run(mcts.root.game_state)
    print(res)

    object_view = lambda x: (
        round(x.UCT(), 2),
        "v" + str(x.visits),
        game.move_to_str(x.move_from_parent),
        "p" + str(x.game_state.get_player()),
    )
    game.draw_state()
    mcts.draw_tree(child_count=3, depth=2, object_view=object_view)
