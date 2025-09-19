import numpy as np
from typing import List


class DenseLayer:
    def __init__(
        self,
        units: int,
        activation: str = "relu",
        weights_initializer: str = "heUniform",
        input_shape: int = None,
    ):
        self.units = units
        self.activation_name = activation
        self.weights_initializer = weights_initializer
        self.input_shape = input_shape

        # Initialize weights and biases (will be set when input shape is known)
        self.weights = None
        self.biases = None

        # Cache for backpropagation
        self.last_input = None
        self.last_z = None
        self.last_output = None

    def initialize_weights(self, input_size: int):
        """Initialize weights and biases based on the specified initializer"""
        if self.weights_initializer == "heUniform":
            limit = np.sqrt(6.0 / input_size)
            self.weights = np.random.uniform(-limit, limit, (input_size, self.units))

        elif self.weights_initializer == "xavier":
            limit = np.sqrt(6.0 / (input_size + self.units))
            self.weights = np.random.uniform(-limit, limit, (input_size, self.units))

        else:  # random
            self.weights = np.random.randn(input_size, self.units) * 0.1

        self.biases = np.zeros((1, self.units))

    def activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "tanh":
            return np.tanh(x)
        elif self.activation_name == "softmax":
            exp_x = np.exp(x)
            print("exp x", exp_x)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function"""
        if self.activation_name == "relu":
            return (x > 0).astype(float)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == "softmax":
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        if self.weights is None:
            self.initialize_weights(inputs.shape[1])

        # Cache inputs for backpropagation
        self.last_input = inputs

        # Linear transformation
        self.last_z = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        print("last z", self.last_z)
        self.last_output = self.activation_function(self.last_z)

        return self.last_output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass through the layer"""
        dz = gradient * self.activation_derivative(self.last_z)

        # Compute gradients w.r.t. weights and biases
        dW = np.dot(self.last_input.T, dz) / self.last_input.shape[0]
        db = np.mean(dz, axis=0, keepdims=True)

        # Compute gradient w.r.t. inputs for previous layer
        dx = np.dot(dz, self.weights.T)

        # Store gradients for weight updates
        self.dW = dW
        self.db = db

        return dx


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers: List[DenseLayer],
        output_layer: DenseLayer,
        learning_rate: float = 0.001,
        loss_type: str = "meanSquaredError",
    ):
        self.layers = hidden_layers + [output_layer]
        self.learning_rate = learning_rate
        self.loss_type = loss_type

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def compute_loss_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str
    ) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions"""
        if loss_type == "categoricalCrossentropy":
            # For softmax + categorical crossentropy, gradient simplifies to:
            return y_pred - y_true
        elif loss_type == "meanSquaredError":
            return 2 * (y_pred - y_true) / y_true.shape[0]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def backward(self, gradient: np.ndarray):
        """Backward pass through all layers"""
        current_gradient = gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def update_weights(self, learning_rate: float):
        """Update weights using computed gradients"""
        for layer in self.layers:
            if hasattr(layer, "dW"):
                layer.weights -= learning_rate * layer.dW
                layer.biases -= learning_rate * layer.db

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int
    ):
        """Train the neural network"""
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}: ", end="")
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            batch_count = 1

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                loss_gradient = self.compute_loss_gradient(
                    y_batch, y_pred, self.loss_type
                )
                self.backward(loss_gradient)

                # Update weights
                self.update_weights(self.learning_rate)

                batch_count += 1
