import numpy as np


class NimNN:
    def __init__(self, input_size, hidden_sizes, output_size_actor, output_size_critic):
        # Initialize weights for each layer
        self.weights = []
        layer_sizes = (
            [input_size] + hidden_sizes + [output_size_actor + output_size_critic]
        )

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            self.weights.append(weight)
        self.output_size_actor = output_size_actor

    def relu(self, x):
        return np.maximum(x, 0)

    def forward_pass(self, x):
        # Forward pass through hidden layers
        for weight in self.weights[:-1]:
            x = self.relu(np.dot(x, weight))

        # Output layer (combined actor and critic)
        combined_output = np.dot(x, self.weights[-1])

        # Splitting actor and critic outputs
        actor_output = combined_output[
            : self.output_size_actor
        ]  # Assuming softmax will be applied later
        critic_output = combined_output[self.output_size_actor :]

        return actor_output, critic_output


# Example parameters
input_size = 3  # For a version of Nim with 3 heaps
hidden_sizes = [64, 64]
output_size_actor = 10  # This would depend on the maximum number of moves available
output_size_critic = 1

# Initialize the neural network
nim_nn = NimNN(input_size, hidden_sizes, output_size_actor, output_size_critic)

# Example input: 3 heaps with 1, 3, and 5 objects
example_input = np.array([1, 3, 5])
actor_output, critic_output = nim_nn.forward_pass(example_input)
print(actor_output, critic_output)
