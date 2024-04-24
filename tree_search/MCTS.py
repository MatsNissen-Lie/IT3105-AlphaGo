import random
import time
from typing import List, Tuple
from config.params import SIMULATIONS, TIME_LIMIT
from game.game_interface import GameInterface
from game.hex import Hex
from game.nim import Nim
from neural_net.anet import ANet
from tree_search.policy import DefaultPolicy, TargetPolicy, TreePlolicy
from tree_search.node import Node
from PrettyPrint import PrettyPrintTree


class MCTS:
    def __init__(
        self,
        game: GameInterface,
        neural_net=None,
        iteration_limit=SIMULATIONS,
        time_limit=TIME_LIMIT,
    ):
        self.root = Node(game)
        self.tree_policy = TreePlolicy()
        self.iteration_limit = iteration_limit
        self.anet = neural_net
        self.time_limit = time_limit
        # self.NN_confidence = 0.3  # Starting value

    def get_root(self):
        return self.root

    def select_node(self) -> "Node":

        node, _ = self.tree_policy.search(self.root)
        return node

    def expand(self, node: Node):
        # Get the possible moves from the game state
        possible_moves = node.game_state.get_legal_moves()
        # random.shuffle(possible_moves)
        # TODO: Vi kan bruke Anet til å velge hvilke moves vi skal legge til i treet!
        for move in possible_moves:
            old_state = node.game_state.clone()
            old_state.make_move(move)
            node.add_child(old_state, move)

    # aka leaf_evaluation
    def simulate(self, node, epsilon):
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
        num_rollouts = 1

        for _ in range(num_rollouts):
            value = self.rollout(node, epsilon)
            total_value += value

        average_value = total_value / num_rollouts
        return average_value

    def rollout(self, node: Node, epsilon=0.1):
        """
        Perform a rollout from the given node using an epsilon-greedy strategy.

        Args:
        node (Node): The node from which the rollout starts.
        epsilon (float): The probability of taking a random action.
        isRandom (bool): Whether to use a random policy for the rollout.
        Returns:
        float: The estimated value of the node.
        """
        # leaf_node: Node = copy.deepcopy(node). Children are not copied
        leaf_node = node
        if self.anet:
            policy = TargetPolicy(self.anet)
            terminal_state = policy(leaf_node, epsilon)
        else:
            policy = DefaultPolicy()
            terminal_state = policy(leaf_node)
        winner = terminal_state.check_win()
        # we minimize for player 2 and maximize for player 1
        return winner if winner != 2 else -1

    def backpropagate(self, node: Node, value):
        while node is not None:
            node.update(value)
            node = node.parent

    def run(self, root_node: Node, epsilon: float = 0.1):
        self.root = root_node
        self.root.parent = None
        sim_start, number_of_simulations = time.time(), 0
        for _ in range(self.iteration_limit):
            # make the loop promt the terminal too continue
            # if _ > 3:
            #     input("Press Enter to continue...")
            #     self.draw_tree()
            node = self.select_node()
            if not node.is_terminal():
                self.expand(node)

            value = self.simulate(node, epsilon)
            self.backpropagate(node, value)
            number_of_simulations += 1
            # if time.time() - sim_start > self.time_limit:
            #     break
        sim_time = time.time() - sim_start
        # print seconds for x simulations
        print(f"Simulations {number_of_simulations} took: {sim_time:.2f} seconds")

        return self.best_node(), self.root.children

    def best_node(self) -> "Node":
        return max(self.root.children, key=lambda x: x.visits)

    def get_move_visits(self) -> List[Tuple[Tuple[int, int], int]]:
        moves = []
        for child in self.root.children:
            moves.append((child.move_from_parent, child.visits))
        return moves

    def draw_tree(
        self,
        child_count=2,
        object_view=lambda x: (
            x.UCT(),
            x.visits,
            "p" + str(x.game_state.get_player()),
        ),
        child_policy=lambda x: x.samle_children(2),
        depth=3,
    ):
        pt = PrettyPrintTree(child_policy, object_view, max_depth=depth)
        if self.root.children:
            pt(self.root)

    def draw_tree_policy(
        self,
        child_count=1,
        depth=10,
    ):
        pt = PrettyPrintTree(
            lambda x: self.tree_policy.get_children_for_draw_tree(x, 1),
            object_view,
            max_depth=depth,
        )
        pt(self.root)


if __name__ == "__main__":
    game = Hex()
    game.go_to_end_game()
    # game = Nim(8, 3)
    anet = ANet()
    mcts = MCTS(game, neural_net=anet, iteration_limit=SIMULATIONS)
    best_node, child_nodes = mcts.run(mcts.get_root(), epsilon=1)

    object_view = lambda x: (
        round(x.UCT(), 2),
        "v" + str(x.visits),
        game.move_to_str(x.move_from_parent),
        # str(x.move_from_parent),
        "p" + str(x.game_state.get_player()),
    )
    game.draw_state()
    # mcts.draw_tree(child_count=2, depth=2, object_view=object_view)
    # mcts.draw_tree_policy()
    mcts.draw_tree(child_count=2, depth=2, object_view=object_view)
