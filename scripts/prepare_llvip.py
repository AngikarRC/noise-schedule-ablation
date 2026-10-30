"""Builds the train/val/test splits and a 512px cache from the raw LLVIP folders.
I have split on scene boundaries rather than randomly, because LLVIP images are
video frames and two frames from one scene are far more alike than two frames
from different scenes, which would leak near-duplicates into the test set.
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

VAL_SCENES = ["17", "18", "25"]
SEED = 20261026


def scene_of(name):
    return name[:2]


def index_scenes(split_dir):
    scenes = {}
    for path in sorted(split_dir.iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            scenes.setdefault(scene_of(path.name), []).append(path.name)
    return scenes


def sample_with_stride(names, quota):
    if quota >= len(names):
        return list(names)
    step = len(names) / quota
    return [names[int(i * step)] for i in range(quota)]


def allocate(scenes, total):
    sizes = {s: len(n) for s, n in scenes.items()}
    pool = sum(sizes.values())
    quotas = {s: max(1, round(total * n / pool)) for s, n in sizes.items()}

    drift = sum(quotas.values()) - total
    order = sorted(quotas, key=lambda s: quotas[s], reverse=True)
    i = 0
    while drift != 0 and i < len(order) * 50:
        s = order[i % len(order)]
        if drift > 0 and quotas[s] > 1:
            quotas[s] -= 1
            drift -= 1
        elif drift < 0 and quotas[s] < sizes[s]:
            quotas[s] += 1
            drift += 1
        i += 1
    return quotas


def process_one(job):
    name, src_vis, src_ir, out_vis, out_ir, resolution = job
    try:
        vis = Image.open(src_vis).convert("RGB")
        ir = Image.open(src_ir).convert("L")
    except Exception as exc:
        return name, f"unreadable: {exc}"

    if vis.size != ir.size:
        return name, f"size mismatch {vis.size} vs {ir.size}"

    resize_square(vis, resolution).save(out_vis, quality=95)
    resize_square(ir, resolution).save(out_ir, quality=95)
    return name, None


def build_split(scenes, total, source_dirs, cache_root, split_name, resolution, workers):
    quotas = allocate(scenes, total)
    records = []

    for scene in sorted(scenes):
        for name in sample_with_stride(scenes[scene], quotas[scene]):
            records.append({"name": name, "scene": scene, "split": split_name})

    out_vis_dir = cache_root / split_name / "visible"
    out_ir_dir = cache_root / split_name / "infrared"
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    out_ir_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (r["name"],
         source_dirs["visible"] / r["name"], source_dirs["infrared"] / r["name"],
         out_vis_dir / r["name"], out_ir_dir / r["name"], resolution)
        for r in records
    ]

    failures = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, err in pool.map(process_one, jobs, chunksize=8):
            done += 1
            if err:
                failures[name] = err
            if done % 250 == 0:
                print(f"    {done}/{len(jobs)}")

    for name, err in failures.items():
        print(f"  skipped {name}: {err}")

    kept = [r for r in records if r["name"] not in failures]
    return kept, quotas


def resize_square(img, resolution):
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((resolution, resolution), Image.LANCZOS)


def write_manifest(path, records):
    with path.open("w") as fh:
        for rec in records:
            fh.write(f"{rec['split']}/{rec['name']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvip", type=Path, required=True,
                    help="path to the LLVIP folder containing visible/ and infrared/")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=60)
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--fixed-batch", type=int, default=4,
                    help="validation images generated at every checkpoint")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    random.seed(SEED)

    src_train = {"visible": args.llvip / "visible" / "train",
                 "infrared": args.llvip / "infrared" / "train"}
    src_test = {"visible": args.llvip / "visible" / "test",
                "infrared": args.llvip / "infrared" / "test"}

    for group in (src_train, src_test):
        for key, path in group.items():
            if not path.is_dir():
                raise SystemExit(f"missing directory: {path}")

    train_scenes_all = index_scenes(src_train["visible"])
    test_scenes_all = index_scenes(src_test["visible"])

    print(f"LLVIP train scenes: {sorted(train_scenes_all)}")
    print(f"LLVIP test scenes : {sorted(test_scenes_all)}")

    missing = [s for s in VAL_SCENES if s not in train_scenes_all]
    if missing:
        raise SystemExit(f"validation scenes not present in train: {missing}")

    val_scenes = {s: train_scenes_all[s] for s in VAL_SCENES}
    fit_scenes = {s: n for s, n in train_scenes_all.items() if s not in VAL_SCENES}

    print(f"\ntrain scenes ({len(fit_scenes)}): {sorted(fit_scenes)}")
    print(f"val scenes   ({len(val_scenes)}): {sorted(val_scenes)}")
    print(f"test scenes  ({len(test_scenes_all)}): {sorted(test_scenes_all)}")

    cache = args.out / "cache"
    args.out.mkdir(parents=True, exist_ok=True)

    res = args.resolution
    w = args.workers
    print(f"\nbuilding train ({args.train_size}) at {res}x{res}, {w} workers ...")
    train, _ = build_split(fit_scenes, args.train_size, src_train, cache, "train", res, w)
    print(f"building val ({args.val_size}) ...")
    val, _ = build_split(val_scenes, args.val_size, src_train, cache, "val", res, w)
    print(f"building test ({args.test_size}) ...")
    test, _ = build_split(test_scenes_all, args.test_size, src_test, cache, "test", res, w)

    write_manifest(args.out / "train.txt", train)
    write_manifest(args.out / "val.txt", val)
    write_manifest(args.out / "test.txt", test)

    fixed = sample_with_stride([r["name"] for r in val], args.fixed_batch)
    (args.out / "val_fixed.txt").write_text(
        "\n".join(f"val/{n}" for n in fixed) + "\n"
    )

    summary = {
        "seed": SEED,
        "resolution": res,
        "split_policy": "scene-disjoint; stride-sampled within scene",
        "train": {"count": len(train), "scenes": sorted(fit_scenes)},
        "val": {"count": len(val), "scenes": sorted(val_scenes)},
        "test": {"count": len(test), "scenes": sorted(test_scenes_all)},
        "fixed_batch": fixed,
    }
    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntrain {len(train)}  val {len(val)}  test {len(test)}")
    print(f"fixed checkpoint batch: {fixed}")

    overlap = ({r["scene"] for r in train} & {r["scene"] for r in val}) | \
              ({r["scene"] for r in train} & {r["scene"] for r in test})
    assert not overlap, f"scene leaked across splits: {overlap}"
    print("no scene appears in more than one split")

    bad = []
    for records in (train, val, test):
        for rec in records:
            for mode in ("visible", "infrared"):
                path = cache / rec["split"] / mode / rec["name"]
                if Image.open(path).size != (res, res):
                    bad.append(str(path))
    if bad:
        raise SystemExit(
            f"{len(bad)} cached files are not {res}x{res}, e.g. {bad[:3]}\n"
            f"Delete {cache} and re-run."
        )
    print(f"all {sum(len(r) for r in (train, val, test)) * 2} cached files are {res}x{res}")

    stale = []
    for split in ("train", "val", "test"):
        wanted = {r["name"] for r in {"train": train, "val": val, "test": test}[split]}
        for mode in ("visible", "infrared"):
            d = cache / split / mode
            stale += [str(p) for p in d.iterdir() if p.name not in wanted]
    if stale:
        print(f"\nnote: {len(stale)} leftover files from a previous run are not "
              f"referenced by any manifest and will be ignored, e.g. {stale[:2]}")

    print(f"\nwrote manifests and {res}x{res} cache under {args.out}")

if __name__ == "__main__":
    main()
