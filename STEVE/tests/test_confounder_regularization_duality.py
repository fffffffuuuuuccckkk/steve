"""Standalone duality tests for latent confounder regularization.

This file intentionally does not require pytest; it can be run directly:

    python tests/test_confounder_regularization_duality.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.test_confounder_regularization import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch

    outputs = run(torch.device(args.device))
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
