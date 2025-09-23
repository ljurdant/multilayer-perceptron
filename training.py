#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np

from src.neuralNetwork import NeuralNetwork, DenseLayer
from data_split import extract_data, format_data, xy_split, train_validation_split


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

        if args.loss not in ["categoricalCrossentropy", "meanSquaredError"]:
            raise ValueError(
                "Loss function must be either 'categoricalCrossentropy' or 'meanSquaredError'"
            )

        return args

    parser = argparse.ArgumentParser(
        description="Train a neural network with specified architecture and parameters."
    )
    parser.add_argument(
        "--train-path", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument(
        "--val-path", type=str, required=False, help="Path to the validation data"
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
    parser.add_argument(
        "--loss",
        type=str,
        default="categoricalCrossentropy",
        help="Loss function to use during training",
    )

    return validate_arguments(parser.parse_args())


if __name__ == "__main__":

    args = parse_args()
    if args.val_path:
        x_train, y_train = xy_split(format_data(extract_data(args.train_path)))
        x_val, y_val = xy_split(format_data(extract_data(args.val_path)))
    else:
        data = format_data(extract_data(args.train_path))
        train_data, val_data = train_validation_split(data, 0.8, args.seed)
        x_train, y_train = xy_split(train_data)
        x_val, y_val = xy_split(val_data)

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
