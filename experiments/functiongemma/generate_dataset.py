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
from experiments.functiongemma.live_context_curriculum import write_v6_dataset
from experiments.functiongemma.production_curriculum import write_v4_dataset
from experiments.functiongemma.recovery_curriculum import write_v5_dataset
from experiments.functiongemma.semantic_context_curriculum import write_v7_dataset

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
        choices=("v3", "v4", "v5", "v6", "v7"),
        default="v3",
        help=(
            "Keep frozen v3, add production-shaped v4, add recovery-focused v5, "
            "add exact-serializer permutation-complete v6 rows, or add broad "
            "exact-runtime semantic v7 rows."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sizes = {
        "train": args.train_size,
        "valid": args.valid_size,
        "test": args.test_size,
    }
    if args.curriculum_version == "v7":
        manifest = write_v7_dataset(args.output_dir, sizes, seed=args.seed)
    elif args.curriculum_version == "v6":
        manifest = write_v6_dataset(args.output_dir, sizes, seed=args.seed)
    elif args.curriculum_version == "v5":
        manifest = write_v5_dataset(args.output_dir, sizes, seed=args.seed)
    elif args.curriculum_version == "v4":
        manifest = write_v4_dataset(args.output_dir, sizes, seed=args.seed)
    else:
        manifest = write_dataset(args.output_dir, sizes, seed=args.seed)
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
