"""Generate the local FunctionGemma AUA candidate-selection curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.functiongemma.curriculum import (
    DEFAULT_SEED,
    DEFAULT_SPLIT_SIZES,
    write_dataset,
)
from experiments.functiongemma.production_curriculum import write_v4_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "functiongemma" / "data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, fictional FunctionGemma learning material. "
            "This command never connects to Android or reads AUA session memory."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument("--valid-size", type=int, default=DEFAULT_SPLIT_SIZES["valid"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument(
        "--curriculum-version",
        choices=("v3", "v4"),
        default="v3",
        help="Keep the frozen v3 default or explicitly add the production-shaped v4 corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sizes = {
        "train": args.train_size,
        "valid": args.valid_size,
        "test": args.test_size,
    }
    manifest = (
        write_v4_dataset(args.output_dir, sizes, seed=args.seed)
        if args.curriculum_version == "v4"
        else write_dataset(args.output_dir, sizes, seed=args.seed)
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "total_records": manifest["total_records"],
                "dataset_sha256": manifest["dataset_sha256"],
                "splits": {
                    name: {
                        "records": item["records"],
                        "groups": item["groups"],
                        "sha256": item["sha256"],
                    }
                    for name, item in manifest["splits"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
