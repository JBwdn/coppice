"""
Command line interface for coppice.
"""
import argparse
from pathlib import Path

import coppicelib
import numpy as np


def parse_cli() -> argparse.Namespace:
    """
    Parse paths to the training data from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("x_path", type=Path)
    parser.add_argument("y_path", type=Path)

    args = parser.parse_args()
    assert args.x_path.exists(), f"{args.x_path} does not exist"
    assert args.y_path.exists(), f"{args.y_path} does not exist"
    assert args.x_path.is_file(), f"{args.x_path} is not a file"
    assert args.y_path.is_file(), f"{args.y_path} is not a file"
    return parser.parse_args()


def main():
    """
    Main script entry point.
    """
    args = parse_cli()

    print("Loading data using numpy...")
    x = np.genfromtxt(args.x_path, delimiter=",")
    y = np.genfromtxt(args.y_path, delimiter=",")

    x = x.astype(np.float32)
    y = y.astype(np.uint32)

    print("Training tree using coppicelib...")
    coppicelib.train_tree_np(x, y)

    print("Training forest using coppicelib...")
    coppicelib.train_forest_np(x, y)


if __name__ == "__main__":
    main()
