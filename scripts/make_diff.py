"""Write a unified diff between the original and generated training scripts.

Using Python avoids PowerShell's `diff` alias for `Compare-Object`.
"""

import argparse
import difflib
from pathlib import Path

BASELINE = "baseline/train_instruct_pix2pix_ORIGINAL.py"
GENERATED = "scripts/finetune.py"
OUT = "report/code_diff.txt"


def main():
    ap = argparse.ArgumentParser(description="Write a unified diff of the training scripts.")
    ap.add_argument("--baseline", default=BASELINE, help="Original training script path.")
    ap.add_argument("--generated", default=GENERATED, help="Generated training script path.")
    ap.add_argument("--out", default=OUT, help="Diff output path.")
    args = ap.parse_args()

    base = Path(args.baseline)
    gen = Path(args.generated)
    for f in (base, gen):
        if not f.exists():
            raise SystemExit(f"missing {f}; run scripts/make_finetune.py --fetch first")

    a = base.read_text(encoding="utf-8").splitlines(keepends=True)
    b = gen.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = list(difflib.unified_diff(a, b, fromfile=str(base), tofile=str(gen), n=3))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")

    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    marked_added = sum(1 for l in b if "# ADDED" in l)
    marked_mod = sum(1 for l in b if "# MODIFIED" in l)

    print(f"wrote {out}")
    print(f"  {added} lines added, {removed} removed, {added + removed} changed in total")
    print(f"  {marked_added} lines marked # ADDED, {marked_mod} marked # MODIFIED")


if __name__ == "__main__":
    main()
