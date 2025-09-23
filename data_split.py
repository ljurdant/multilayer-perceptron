#!/usr/bin/env python3
import numpy as np
import pandas as pd


def extract_data(data_path: str) -> np.ndarray:
    df = pd.read_csv(data_path)
    # Perform feature extraction on the DataFrame
    return df.to_numpy()


def train_validation_split(data, split_ratio, seed):
    np.random.seed(seed)
    np.random.shuffle(
        data,
    )
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data


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


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Extract and split data into features and labels."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the input data for splitting",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="Train-validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    data = extract_data(args.data_path)
    train_data, val_data = train_validation_split(data, args.split_ratio, args.seed)

    # Save csvs train and validation data
    df = pd.DataFrame(train_data)
    df.to_csv("train_data.csv", index=False, header=False)
    df = pd.DataFrame(val_data)
    df.to_csv("val_data.csv", index=False, header=False)
