"""Create the report figure from saved inference trajectories."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SAMPLE = "190001"
OUT = Path("report/figures/fig6_trajectories.png")


def row(ax_row, run, info):
    items = [i for i in info["images"] if i["sample"].startswith(SAMPLE)]
    items.sort(key=lambda i: i["sampler_step"])
    for ax, item in zip(ax_row, items):
        img = Image.open(Path(f"trajectories/run{run}") / item["file"])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        scale = item.get("display_scale", 1.0)
        tag = f"t = {item['timestep']}"
        if scale and scale > 1.5:
            tag += f"\n(shown /{scale:.0f})"
        ax.set_title(tag, fontsize=8)


def main():
    infos = {r: json.loads(Path(f"trajectories/run{r}/_trajectory_info.json").read_text())
             for r in "AB"}
    n = len(infos["A"]["capture_steps"])
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.4))
    for i, r in enumerate("AB"):
        row(axes[i], r, infos[r])
        axes[i][0].set_ylabel(f"Run {r}", fontsize=11)
    fig.suptitle(f"Reverse-process samples x_t, test image {SAMPLE}.jpg", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
