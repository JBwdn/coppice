# coppice

A simple decision tree and random forest library using a Rust backend.

## Installation

```bash
pip install coppice
```

## Usage

### Python
```python
import numpy as np

from coppice import _coppice

x = ... # [w, n]
y = ... # [n]

x = x.astype(np.float32)
y = y.astype(np.uint32)

_coppice.train_tree_np(x, y, max_depth=5)

_coppice.train_forest_np(x, y, max_depth=5, num_trees=10)
```

### CLI

```bash
usage: coppice_ [-h] [--model-type {tree,forest}] [--num-trees NUM_TREES]
                [--max-depth MAX_DEPTH]
                x-path y-path

positional arguments:
  x-path
  y-path

options:
  -h, --help            show this help message and exit
  --model-type {tree,forest}
  --num-trees NUM_TREES
  --max-depth MAX_DEPTH
```

## Development

```bash
git clone git@github.com:JBwdn/coppice.git
cd coppice
pip install -e ".[dev]"
maturin develop
pre-commit install
```
