#!/usr/bin/env python3

import argparse
from neuralNetwork import NeuralNetwork
from training import extract_data, format_data, xy_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained neural network model."
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the input data for predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load the trained model
    model = NeuralNetwork.load(args.model_path)
    print("Model loaded successfully.")
    # Load input data

    data = extract_data(args.data_path)
    data = format_data(data)
    X, y = xy_split(data)

    # Make predictions
    predictions = model.predict(X)
    y = model.one_hot_encode(y, predictions.shape[1])
    loss = model.compute_loss(y, predictions, "categoricalCrossentropy")
    accuracy = model.compute_accuracy(y, predictions)

    print(f"Predictions made successfully. Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save predictions
    # model.save_predictions(predictions, args.output_data)

    # print("Predictions saved successfully.")
