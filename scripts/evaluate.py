"""Compare Run 0, Run A, and Run B predictions with held-out infrared images.

Reports SSIM, PSNR, luminance error, and contrast ratio.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def score_run(pred_dir, data_root, manifest):
    root = Path(data_root)
    rels = [l.strip() for l in (root / manifest).read_text().splitlines() if l.strip()]

    rows = []
    for rel in rels:
        split, name = rel.split("/", 1)
        pred_path = Path(pred_dir) / name
        if not pred_path.exists():
            continue

        truth = load_gray(root / "cache" / split / "infrared" / name)
        pred = load_gray(pred_path)
        if pred.shape != truth.shape:
            pred = np.array(
                Image.open(pred_path).convert("L").resize(truth.shape[::-1], Image.LANCZOS),
                dtype=np.float64,
            )

        rows.append({
            "name": name,
            "ssim": structural_similarity(truth, pred, data_range=255.0),
            "psnr": peak_signal_noise_ratio(truth, pred, data_range=255.0),
            "pred_mean": pred.mean(),
            "pred_std": pred.std(),
            "true_mean": truth.mean(),
            "true_std": truth.std(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, None

    summary = {
        "n": len(df),
        "ssim_mean": df["ssim"].mean(),
        "ssim_std": df["ssim"].std(),
        "psnr_mean": df["psnr"].mean(),
        "psnr_std": df["psnr"].std(),
        "luminance_mean_error": (df["pred_mean"] - df["true_mean"]).mean(),
        "luminance_mean_abs_error": (df["pred_mean"] - df["true_mean"]).abs().mean(),
        "contrast_ratio": (df["pred_std"] / df["true_std"]).mean(),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate the baseline and both fine-tuned runs.")
    ap.add_argument("--data", default="data", help="Prepared data folder.")
    ap.add_argument("--manifest", default="test.txt", help="Test manifest filename.")
    ap.add_argument("--run0", default="outputs/run0", help="Run 0 predictions folder.")
    ap.add_argument("--runA", default="outputs/runA/final", help="Run A predictions folder.")
    ap.add_argument("--runB", default="outputs/runB/final", help="Run B predictions folder.")
    ap.add_argument("--out", default="report", help="Report output folder.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, path in [("Run 0", args.run0), ("Run A", args.runA), ("Run B", args.runB)]:
        if not Path(path).is_dir():
            print(f"{label:<7} skipped ({path} not found)")
            continue
        df, summary = score_run(path, args.data, args.manifest)
        if summary is None:
            print(f"{label:<7} skipped (no matching predictions in {path})")
            continue
        results[label] = summary
        df.to_csv(out / f"per_image_{label.replace(' ', '').lower()}.csv", index=False)
        print(f"{label:<7} scored {summary['n']} images")

    if not results:
        raise SystemExit("nothing to score - generate predictions first")

    print()
    header = f"{'run':<8}{'SSIM':>16}{'PSNR (dB)':>16}{'lum. err':>12}{'contrast':>11}"
    print(header)
    print("-" * len(header))
    for label, s in results.items():
        print(f"{label:<8}"
              f"{s['ssim_mean']:>10.4f} ±{s['ssim_std']:<5.3f}"
              f"{s['psnr_mean']:>10.2f} ±{s['psnr_std']:<5.2f}"
              f"{s['luminance_mean_error']:>12.2f}"
              f"{s['contrast_ratio']:>11.3f}")

    print("\nluminance err: mean(predicted - true) grey level, 0 is perfect")
    print("contrast     : mean(predicted sd / true sd), 1.0 is perfect")

    table = pd.DataFrame(results).T
    table.index.name = "run"
    table.to_csv(out / "metrics.csv")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'metrics.csv'}")

if __name__ == "__main__":
    main()
