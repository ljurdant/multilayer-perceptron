import numpy as np
from typing import List
import pickle
import matplotlib.pyplot as plt


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
            x = np.asarray(x, dtype=np.float64)
            exp_x = np.exp(x)
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
        self.dW = np.asarray(dW, dtype=np.float64)
        self.db = np.asarray(db, dtype=np.float64)

        return dx


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers: List[DenseLayer],
        output_layer: DenseLayer,
        learning_rate: float = 0.001,
        loss_type: str = "categoricalCrossentropy",
    ):
        self.layers = hidden_layers + [output_layer]
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.training_history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        self.val_stagnation_counter = 0
        self.max_val_stagnation = 10
        self.stagnation_counter = 0
        self.max_stagnation = 5

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str
    ) -> float:
        """Compute loss based on loss type"""
        if loss_type == "categoricalCrossentropy":
            # Clip predictions to prevent log(0)
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        elif loss_type == "meanSquaredError":
            return np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy"""
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)

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
        if self.stagnation_counter >= self.max_stagnation:
            learning_rate *= 0.5
            self.stagnation_counter = 0
            print(
                f"\nStagnation detected. Reducing learning rate to {learning_rate:.6f}"
            )
            print("Continuing training...\n")
        for layer in self.layers:
            if hasattr(layer, "dW"):
                layer.weights -= learning_rate * layer.dW
                layer.biases -= learning_rate * layer.db

    def one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        return one_hot

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        """Train the neural network"""
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))
        self.stagnation_counter = 0

        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            batch_count = 1
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Convert true values to one-hot if using categorical crossentropy
                y_batch_one_hot = self.one_hot_encode(y_batch, self.layers[-1].units)

                # Backward pass
                loss_gradient = self.compute_loss_gradient(
                    y_batch_one_hot, y_pred, self.loss_type
                )
                self.backward(loss_gradient)

                # Update weights
                self.update_weights(self.learning_rate)

                loss = self.compute_loss(y_batch_one_hot, y_pred, self.loss_type)
                accuracy = self.compute_accuracy(y_batch_one_hot, y_pred)

                print(
                    f"\repoch {epoch + 1:0{len(str(epochs))}d}/{epochs}: batch {batch_count:0{len(str(n_batches))}d}/{n_batches} loss={loss:.4f} accuracy={accuracy:.4f}",
                    end="",
                )
                epoch_loss += loss
                epoch_accuracy += accuracy

                batch_count += 1

            # Validation
            y_val_one_hot = self.one_hot_encode(y_val, self.layers[-1].units)
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val_one_hot, val_pred, self.loss_type)
            val_accuracy = self.compute_accuracy(y_val_one_hot, val_pred)

            epoch_accuracy /= n_batches
            epoch_loss /= n_batches

            # Store history
            self.training_history["loss"].append(epoch_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["accuracy"].append(epoch_accuracy)
            self.training_history["val_accuracy"].append(val_accuracy)

            if epoch > 1:
                if (
                    self.training_history["val_loss"][-1]
                    >= self.training_history["val_loss"][-2]
                ):
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            print(
                f"\repoch {epoch + 1:0{len(str(epochs))}d}/{epochs}: batch {n_batches}/{n_batches} loss={epoch_loss:.4f} accuracy={epoch_accuracy:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}",
            )

            if self.stagnation_counter >= self.max_stagnation:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} due to stagnation in validation loss."
                )
                break

    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            "hidden_layers": [],
            "output_layer": {},
            "training_history": self.training_history,
        }

        for layer in self.layers[:-1]:
            layer_data = {
                "units": layer.units,
                "activation": layer.activation_name,
                "weights_initializer": layer.weights_initializer,
                "weights": layer.weights,
                "biases": layer.biases,
            }
            model_data["hidden_layers"].append(layer_data)
        output_layer = self.layers[-1]

        output_layer_data = {
            "units": output_layer.units,
            "activation": output_layer.activation_name,
            "weights_initializer": output_layer.weights_initializer,
            "weights": output_layer.weights,
            "biases": output_layer.biases,
        }
        model_data["output_layer"] = output_layer_data

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def plot(self):
        """Plot training and validation loss and accuracy"""

        epochs = range(1, len(self.training_history["loss"]) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.training_history["loss"], label="Training Loss")
        plt.plot(epochs, self.training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.training_history["accuracy"], label="Training Accuracy")
        plt.plot(
            epochs, self.training_history["val_accuracy"], label="Validation Accuracy"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    @classmethod
    def load(cls, filepath: str):
        """Load a trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        hidden_layers = []
        for layer_data in model_data["hidden_layers"]:
            layer = DenseLayer(
                units=layer_data["units"],
                activation=layer_data["activation"],
                weights_initializer=layer_data["weights_initializer"],
            )
            layer.weights = layer_data["weights"]
            layer.biases = layer_data["biases"]
            hidden_layers.append(layer)
        output_layer_data = model_data["output_layer"]
        output_layer = DenseLayer(
            units=output_layer_data["units"],
            activation=output_layer_data["activation"],
            weights_initializer=output_layer_data["weights_initializer"],
        )
        output_layer.weights = output_layer_data["weights"]
        output_layer.biases = output_layer_data["biases"]

        model = cls(hidden_layers, output_layer)
        model.training_history = model_data["training_history"]
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X)
