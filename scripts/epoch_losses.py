"""Average per-step training losses into one value per epoch.

The source log remains available for plotting and checking training stability.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Convert step losses to per-epoch averages.")
    ap.add_argument("--in", dest="source", type=Path, required=True, help="Per-step loss CSV.")
    ap.add_argument("--out", dest="target", type=Path, required=True, help="Per-epoch output CSV.")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} not found - run training first")

    losses = defaultdict(list)
    with args.source.open() as fh:
        for row in csv.DictReader(fh):
            losses[int(row["epoch"])].append(float(row["train_loss"]))

    if not losses:
        raise SystemExit(f"{args.source} has no rows")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss"])
        for epoch in sorted(losses):
            mean = sum(losses[epoch]) / len(losses[epoch])
            writer.writerow([epoch + 1, f"{mean:.6f}"])

    print(f"{args.source.name} -> {args.target.name}")
    for epoch in sorted(losses):
        values = losses[epoch]
        print(f"  epoch {epoch + 1}: {sum(values) / len(values):.6f} "
              f"over {len(values)} steps")

if __name__ == "__main__":
    main()
