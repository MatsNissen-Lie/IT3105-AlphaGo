import random
from neural_net.anet import ANet
from tree_search.node import Node
from game.game_interface import GameInterface
import numpy as np
from typing import List, Tuple


class TreePlolicy:

    def __init__(self):
        pass

    def get_child(self, node: Node) -> Node:
        """
        Select the child node with the highest UCT value.

        Returns
        -------
        child_node: Node
            The child node with the highest UCT value.
        """
        if node.children == []:
            raise ValueError("The node has no children.")

        optimizer = max if node.get_state().get_player() == 1 else min
        child_node = optimizer(node.children, key=lambda x: x.UCT())
        return child_node

    def get_children_orderd_by_UCT(self, node: Node) -> List[Node]:
        """
        Select the child node with the highest UCT value.

        Returns
        -------

            The child nodes ordered by the highest UCT value.
        """
        if node.children == []:
            return []

        reversed = node.get_state().get_player() == 2
        children = sorted(node.children, key=lambda x: x.UCT(), reverse=reversed)
        return children

    def get_children_for_draw_tree(
        self, node: Node, count: int = 1, prob_extra_child: float = 0.25
    ) -> List[Node]:
        children = self.get_children_orderd_by_UCT(node)
        # 25% change of adding and extra child
        if len(children) > 2 and random.random() < prob_extra_child:
            return children[: count + 1]
        return children[:count]

    def search(self, root_node: Node) -> Tuple[Node, int]:
        """
        Select the child node with the highest UCT value.

        Returns
        -------
        child_node: Node
            The child node with the highest UCT value.
        """
        node = root_node
        depth = 1
        while node.children != []:
            node = self.get_child(node)
            depth += 1
        return node, depth


class DefaultPolicy:
    """
    The DefaultPolicy class is used to represent the default policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to evaluate the leaf nodes. As the default policy, the target policy is
    used, since we are using on-policy Monte Carlo Tree Search.
    """

    def __call__(self, curr_node: Node) -> GameInterface:
        """
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        curr_state : state of the leaf node

        return
        ------
        curr_state : terminal state of the game
        """

        # TODO: The default policy should not add nodes to the tree. It should only simulate the game.
        curr_state = curr_node.game_state.clone()
        while not curr_state.is_terminal():
            possible_moves = curr_state.get_legal_moves()
            move = random.choice(possible_moves)
            curr_state.make_move(move)
        return curr_state


class TargetPolicy:
    """
    The TargetPolicy class is used to represent the target policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to select the best child node to explore. As the target policy, the actor
    is used, since we are using on-policy Monte Carlo Tree Search.
    """

    def __init__(self, anet: ANet):
        self.anet = anet

    # TODO: The target policy should NOT add nodes to the tree.
    def __call__(self, curr_node: Node, epsilon: float = 0.1) -> GameInterface:
        """
        Using the actor to select the best child node to explore. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        curr_state : state of the leaf node
        possible_moves : list of possible moves

        return
        ------
        move : best move to explore
        """

        # TODO: this doesnt work for some reason

        curr_state = curr_node.game_state.clone()
        while not curr_state.is_terminal():
            if random.random() < epsilon:
                possible_moves = curr_state.get_legal_moves()
                move = random.choice(possible_moves)
            else:
                pred = self.anet.predict(curr_state.get_nn_input())
                move = curr_state.get_move_from_nn_output(pred)
            curr_state.make_move(move)
        return curr_state


class TargetPolicyTest:

    def __init__(self, anet: ANet):
        self.anet = anet

    def __call__(self, curr_node: Node, epsilon: float = 0.1) -> GameInterface:

        curr_state = curr_node.game_state.clone()
        while not curr_state.is_terminal():

            # print(f"\n\nPlayer {curr_state.get_player()}")
            if random.random() < epsilon:
                possible_moves = curr_state.get_legal_moves()
                move = random.choice(possible_moves)
            else:
                time_start = time.time()
                pred = self.anet.predict(curr_state.get_nn_input(), verbose=1)
                # time taken in milliseconds
                time_taken = (time.time() - time_start) * 1000
                print(f"Time taken: {time_taken:.2f}ms")
                # print preds rounded to 2 decimals in a four by four matrix
                # print(np.round(pred.reshape(4, 4), 2))
                move = curr_state.get_move_from_nn_output(pred)

            curr_state.make_move(move)
            # curr_state.draw_state()
        return curr_state


if __name__ == "__main__":
    import time
    from neural_net.anet import load_model
    from game.hex import Hex

    game_size = 4
    node = Node(Hex(game_size))

    anet = load_model("tournament3", 0, "hex", board_size=game_size)

    startTime = time.time()
    policy = TargetPolicyTest(anet)
    state = policy(node, 0)
    taken_time = time.time() - startTime
    print(f"Time taken: {taken_time:.2f}s")
    print(f"Winner: {state.check_win()}")
