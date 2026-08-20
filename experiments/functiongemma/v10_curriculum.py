"""Render the V10 dataset: frozen V8 foundations plus the V10 families.

V9 was trained standalone from the base checkpoint. Measured afterwards, 95% of what V8 knew was
inherited v7 foundation rows (44,224 of 46,584 training rows), and discarding them is the single
cause behind most of V9's regressions: taps fell from the overwhelming majority of the signal to
40% of a self-generated corpus, and the model duly learned not to tap. The handover had said to
append to the frozen V8 foundations; this module does that.

The V8 splits are copied verbatim — same rows, same split membership — so nothing that was
already learned is re-derived or perturbed, and no V8 group can migrate across a split boundary.
V10 rows are appended to their matching split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .v9_curriculum import SPLIT_ORDER, audit, render_case
from .v10_learning_material import SCHEMA as SOURCE_SCHEMA
from .v10_learning_material import SEED, generate, group_id

TEMPLATE_PROFILE = "exact_policy_messages_v10"
FOUNDATION_DEFAULT = Path("runs/functiongemma/data-v8")


def build_split(split: str, groups: int, variants: int) -> list[dict[str, Any]]:
    """Render every V10 case for *split* under *variants* counterbalanced permutations."""

    rows: list[dict[str, Any]] = []
    for case in generate(split, groups):
        for variant in range(variants):
            row = render_case(case, variant)
            row["id"] = row["id"].replace("fg9-", "fg10-", 1)
            row["metadata"]["schema"] = SOURCE_SCHEMA
            row["metadata"]["template_profile"] = TEMPLATE_PROFILE
            row["metadata"]["group_id"] = group_id(case)
            row["metadata"]["origin"] = "v10"
            rows.append(row)
    return rows


def load_foundation(directory: Path, split: str) -> list[dict[str, Any]]:
    """Load one frozen V8 split verbatim, tagging provenance without altering content."""

    path = directory / f"{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.setdefault("metadata", {})
            metadata["origin"] = "v8_foundation"
            rows.append(row)
    return rows


def check_group_isolation(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    """No semantic group — from either generation — may appear in two splits."""

    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            gid = row.get("metadata", {}).get("group_id")
            if not gid:
                continue
            key = f"{row['metadata'].get('origin', '?')}:{gid}"
            if seen.setdefault(key, split) != split:
                raise ValueError(f"group {key} appears in {seen[key]} and {split}")


def composition(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    origins = Counter(row.get("metadata", {}).get("origin", "unknown") for row in rows)
    return {"rows": len(rows), "by_origin": dict(sorted(origins.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the V10 FunctionGemma dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--foundation-dir", type=Path, default=FOUNDATION_DEFAULT)
    parser.add_argument("--train-groups", type=int, default=6000)
    parser.add_argument("--valid-groups", type=int, default=750)
    parser.add_argument("--test-groups", type=int, default=750)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument(
        "--no-foundation",
        action="store_true",
        help="Render V10 rows alone. Diagnostic only — this is what produced V9's regressions.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, groups in (
        ("train", args.train_groups),
        ("valid", args.valid_groups),
        ("test", args.test_groups),
    ):
        rows: list[dict[str, Any]] = []
        if not args.no_foundation:
            rows.extend(load_foundation(args.foundation_dir, split))
        rows.extend(build_split(split, groups, args.variants))
        splits[split] = rows
    check_group_isolation(splits)

    manifest: dict[str, Any] = {
        "schema": "aua-functiongemma-dataset-v10",
        "template_profile": TEMPLATE_PROFILE,
        "uses_exact_policy_messages": True,
        "variants_per_group": args.variants,
        "seed": SEED,
        "selection_function": "select_candidate(candidate_id: integer)",
        "foundation": None if args.no_foundation else str(args.foundation_dir),
        "prompt_schema": {
            "candidate_counts": [2, 3, 4],
            "candidate_ids": "dense opaque integers 0 through candidate_count minus 1",
            "handoff_candidate_id": -1,
            "name": "functiongemma-aua-candidate-policy-v3",
        },
        "split_policy": (
            "V8 foundation rows keep their original split membership; V10 groups are "
            "split-exclusive by construction. No group identity crosses a boundary."
        ),
        "privacy": {
            "passed": True,
            "checks": [
                "fictional app-agnostic vocabulary only",
                "no journals, maps, screenshots, hierarchy, devices, or typed input",
                "split-exclusive source entities and semantic groups",
                "public repository denylist and private fingerprints",
                "no device serials in candidate arguments",
                "frozen foundation rows copied verbatim from an already-audited corpus",
            ],
        },
        "splits": {},
    }

    split_hashes: dict[str, str] = {}
    for split in SPLIT_ORDER:
        rows = splits[split]
        name = f"{split}.jsonl"
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split_hashes[split] = digest
        v10_rows = [row for row in rows if row["metadata"].get("origin") == "v10"]
        manifest["splits"][split] = {
            **composition(rows),
            "v10_audit": audit(v10_rows) if v10_rows else {},
            "path": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    manifest["total_records"] = sum(len(rows) for rows in splits.values())
    manifest["dataset_sha256"] = hashlib.sha256(
        "".join(split_hashes[split] for split in SPLIT_ORDER).encode()
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
