"""
Command line interface for coppice.
"""
import argparse
from pathlib import Path

import numpy as np

from coppice import _coppice  # type: ignore


def parse_cli() -> argparse.Namespace:
    """
    Parse paths to the training data from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("x_path", metavar="x-path", type=Path)
    parser.add_argument("y_path", metavar="y-path", type=Path)
    parser.add_argument(
        "--model-type", type=str, default="tree", choices=["tree", "forest"]
    )
    parser.add_argument("--num-trees", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=100)

    args = parser.parse_args()
    assert args.x_path.exists(), f"{args.x_path} does not exist"
    assert args.x_path.is_file(), f"{args.x_path} is not a file"
    assert args.x_path.suffix == ".csv", f"{args.x_path} is not a csv file"

    assert args.y_path.exists(), f"{args.y_path} does not exist"
    assert args.y_path.is_file(), f"{args.y_path} is not a file"
    assert args.y_path.suffix == ".csv", f"{args.y_path} is not a csv file"

    assert args.model_type in [
        "tree",
        "forest",
    ], f"{args.model_type} is not a valid model type"

    assert args.num_trees > 0, f"{args.num_trees} must be >= 1"
    assert args.max_depth > 0, f"{args.max_depth} must be >= 1"

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

    match args.model_type:
        case "tree":
            print("Training tree using coppicelib...")
            _coppice.train_tree_np(x, y, args.max_depth)
        case "forest":
            print("Training forest using coppicelib...")
            _coppice.train_forest_np(x, y, args.max_depth, args.num_trees)
        case _:
            raise ValueError(f"Invalid model type: {args.model_type}")


if __name__ == "__main__":
    main()
