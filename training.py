#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np

from neuralNetwork import NeuralNetwork, DenseLayer


def parse_args():
    def validate_arguments(args):
        # Validate layer sizes
        if any(layer <= 0 for layer in args.layers):
            raise ValueError("All layer sizes must be positive integers")

        # Validate epochs
        if args.epochs <= 0:
            raise ValueError("Epochs must be a positive integer")

        # Validate batch size
        if args.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        # Validate learning rate
        if args.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        return args

    parser = argparse.ArgumentParser(
        description="Train a neural network with specified architecture and parameters."
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer sizes for the neural network architecture (e.g., --layer 24 24 24)",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return validate_arguments(parser.parse_args())


def extract_data(data_path: str) -> np.ndarray:
    df = pd.read_csv(data_path)
    # Perform feature extraction on the DataFrame
    return df.to_numpy()


def format_data(data: np.ndarray) -> np.ndarray:
    def zscore_normalize(column: np.ndarray) -> np.ndarray:
        mean = np.mean(column)
        std = np.std(column)
        return (column - mean) / std

    for i in range(data.shape[1]):
        if i == 1:
            data[:, i] = (data[:, i] == "M").astype(int)
        else:
            data[:, i] = zscore_normalize(data[:, i].astype(float))

    return data


def xy_split(data):
    X = np.concatenate((data[:, 0].reshape(-1, 1), data[:, 2:]), axis=1)
    y = data[:, 1]
    return X, y


def get_data(data_path: str, seed):

    def train_validation_split(data, split_ratio=0.8):
        np.random.seed(seed)
        np.random.shuffle(
            data,
        )
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        val_data = data[split_index:]

        return train_data, val_data

    data_path = args.data_path
    data = extract_data(data_path)
    data = format_data(data)
    train_data, val_data = train_validation_split(data)
    x_train, y_train = xy_split(train_data)
    x_val, y_val = xy_split(val_data)
    return x_train, y_train, x_val, y_val


if __name__ == "__main__":

    args = parse_args()
    x_train, y_train, x_val, y_val = get_data(args.data_path, seed=args.seed)

    model = NeuralNetwork(
        hidden_layers=[
            DenseLayer(units=size, activation="relu") for size in args.layers
        ],
        output_layer=DenseLayer(units=2, activation="softmax"),
        learning_rate=args.learning_rate,
    )

    model.fit(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model.save("model.pkl")
    model.plot()
