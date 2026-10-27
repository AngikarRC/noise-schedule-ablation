"""Aggregate the step-level loss log into the per-epoch CSV the assignment asks for.

finetune.py logs every optimizer step, which is more useful for plotting and
for spotting instability. The submission wants columns `epoch, train_loss`,
so this collapses the former into the latter rather than having the training
loop write two files.

    python scripts/epoch_losses.py --in logs/runA_loss.csv --out logs/runA_epoch_loss.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="source", type=Path, required=True)
    ap.add_argument("--out", dest="target", type=Path, required=True)
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
            # 1-indexed: the training loop counts from 0, the report counts epochs.
            writer.writerow([epoch + 1, f"{mean:.6f}"])

    print(f"{args.source.name} -> {args.target.name}")
    for epoch in sorted(losses):
        values = losses[epoch]
        print(f"  epoch {epoch + 1}: {sum(values) / len(values):.6f} "
              f"over {len(values)} steps")


if __name__ == "__main__":
    main()
