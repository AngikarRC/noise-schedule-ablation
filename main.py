"""Runs the whole experiment in order and prints every command before it runs,
so the terminal log doubles as the record of what produced the results.
I have kept each stage runnable on its own, so this file is a convenience
rather than something the pipeline depends on.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

SEED = "20261026"
RESOLUTION = "512"

TRAIN_PAIRS = 2000
EFFECTIVE_BATCH = 4
STEPS_PER_EPOCH = TRAIN_PAIRS // EFFECTIVE_BATCH
EPOCHS = 5
STEPS = STEPS_PER_EPOCH * EPOCHS
CKPT_EVERY = STEPS_PER_EPOCH
CHECKPOINTS = list(range(CKPT_EVERY, STEPS + 1, CKPT_EVERY))

MODEL = "timbrooks/instruct-pix2pix"

EXPECTED_CHECKSUMS = {"A": "40873f7067e7f82d", "B": "6fa947b412c6432e"}


def train_command(run_mode):
    return [
        PY, "scripts/finetune.py",
        "--run_mode", run_mode,
        "--output_dir", f"outputs/run{run_mode}",
        "--loss_csv", f"logs/run{run_mode}_loss.csv",
        "--pretrained_model_name_or_path", MODEL,
        "--llvip_data", "data",
        "--resolution", RESOLUTION,
        "--center_crop",
        "--train_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(STEPS),
        "--checkpointing_steps", str(CKPT_EVERY),
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--mixed_precision", "fp16",
        "--gradient_checkpointing",
        "--lora_rank", "16",
        "--lora_alpha", "16",
        "--dataloader_num_workers", "0",
        "--seed", SEED,
    ]


def infer_command(run_mode, lora, out, manifest=None):
    cmd = [PY, "scripts/infer.py", "--run_mode", run_mode,
           "--out", out, "--seed", SEED]
    if lora:
        cmd += ["--lora", lora]
    if manifest:
        cmd += ["--manifest", manifest]
    return cmd


def build_stages():
    stages = [
        ("env", "Verify GPU and benchmark",
         [[PY, "scripts/check_env.py"]], "logs/env.json"),

        ("data", "Build scene-disjoint splits and the 512px cache",
         [[PY, "scripts/prepare_llvip.py", "--llvip", "LLVIP", "--out", "data",
           "--resolution", RESOLUTION, "--train-size", "2000",
           "--val-size", "60", "--test-size", "100"]], "data/train.txt"),

        ("schedule", "Verify the schedule and plot log-SNR",
         [[PY, "scripts/schedule.py"]], "schedules/checksums.txt"),

        ("patch", "Fetch the pinned baseline and generate finetune.py",
         [[PY, "scripts/make_finetune.py", "--fetch"]], "scripts/finetune.py"),

        ("diff", "Write the baseline-to-finetune diff for the report",
         [[PY, "scripts/make_diff.py"]], "report/code_diff.txt"),

        ("train-a", f"Run A - native schedule ({EPOCHS} epochs, {STEPS} steps)",
         [train_command("A")], f"outputs/runA/checkpoint-{STEPS}"),

        ("train-b", f"Run B - alpha_bar ^ 2.35 ({EPOCHS} epochs, {STEPS} steps)",
         [train_command("B")], f"outputs/runB/checkpoint-{STEPS}"),

        ("epoch-logs", "Collapse step losses into the epoch,train_loss CSVs",
         [[PY, "scripts/epoch_losses.py",
           "--in", f"logs/run{r}_loss.csv",
           "--out", f"logs/run{r}_epoch_loss.csv"] for r in ("A", "B")],
         "logs/runB_epoch_loss.csv"),

        ("run0", "Run 0 - pretrained baseline, no fine-tuning",
         [infer_command("0", None, "outputs/run0")], "outputs/run0/_run_info.json"),

        ("final-a", "Final inference, Run A",
         [infer_command("A", f"outputs/runA/checkpoint-{STEPS}", "outputs/runA/final")],
         "outputs/runA/final/_run_info.json"),

        ("final-b", "Final inference, Run B",
         [infer_command("B", f"outputs/runB/checkpoint-{STEPS}", "outputs/runB/final")],
         "outputs/runB/final/_run_info.json"),
    ]

    for run in ("A", "B"):
        cmds = [
            infer_command(run, f"outputs/run{run}/checkpoint-{s}",
                          f"outputs/run{run}/ckpt-{s}", manifest="val_fixed.txt")
            for s in CHECKPOINTS
        ]
        stages.append((
            f"ckpt-{run.lower()}",
            f"Samples at {len(CHECKPOINTS)} epoch boundaries, Run {run}",
            cmds,
            f"outputs/run{run}/ckpt-{CHECKPOINTS[-1]}/_run_info.json",
        ))

    for run in ("A", "B"):
        stages.append((
            f"traj-{run.lower()}",
            f"Inference trajectories, Run {run}",
            [[PY, "scripts/trajectory.py", "--run_mode", run,
              "--lora", f"outputs/run{run}/checkpoint-{STEPS}",
              "--out", f"trajectories/run{run}", "--seed", SEED]],
            f"trajectories/run{run}/_trajectory_info.json",
        ))

    stages.append((
        "eval", "Score Run 0, A and B",
        [[PY, "scripts/evaluate.py",
          "--runA", "outputs/runA/final", "--runB", "outputs/runB/final"]],
        None,
    ))

    return stages


def assert_single_variable():
    a, b = train_command("A"), train_command("B")
    if len(a) != len(b):
        raise SystemExit("Run A and Run B commands have different lengths")

    differences = {(x, y) for x, y in zip(a, b) if x != y}
    expected = {
        ("A", "B"),
        ("outputs/runA", "outputs/runB"),
        ("logs/runA_loss.csv", "logs/runB_loss.csv"),
    }
    if differences != expected:
        raise SystemExit(
            "Run A and Run B differ in unexpected places, so the ablation "
            f"would not be single-variable:\n  {differences - expected}"
        )


def show(cmd):
    parts = []
    for c in cmd:
        c = str(c)
        if c == PY:
            c = "python"
        parts.append(f'"{c}"' if " " in c else c)
    return " ".join(parts)


def check_prerequisites(stage_names):
    problems = []
    if "data" in stage_names and not (ROOT / "LLVIP" / "visible" / "train").is_dir():
        problems.append(
            "LLVIP/visible/train not found. Download LLVIP from "
            "https://bupt-ai-cz.github.io/LLVIP/ and unpack it to LLVIP/"
        )
    needs_data = {"train-a", "train-b", "run0", "final-a", "final-b", "eval"}
    if needs_data & set(stage_names) and not (ROOT / "data" / "train.txt").exists():
        if "data" not in stage_names:
            problems.append("data/train.txt not found - run the 'data' stage first")
    return problems


def verify_checksums():
    import json

    print("\nSchedule checksums actually used")
    ok = True
    for run, expected in EXPECTED_CHECKSUMS.items():
        path = ROOT / f"outputs/run{run}/schedule/schedule_run{run}.json"
        if not path.exists():
            print(f"  run {run}: no record (stage not run)")
            continue
        got = json.loads(path.read_text())["checksum"]
        mark = "ok" if got == expected else "MISMATCH"
        print(f"  run {run}: {got}  expected {expected}  {mark}")
        ok &= got == expected

    a = ROOT / "outputs/runA/schedule/schedule_runA.json"
    b = ROOT / "outputs/runB/schedule/schedule_runB.json"
    if a.exists() and b.exists():
        ha = json.loads(a.read_text())["checksum"]
        hb = json.loads(b.read_text())["checksum"]
        if ha == hb:
            print("\n  Run A and Run B used the SAME schedule. The transform did "
                  "not apply and the comparison is void.")
            ok = False
        else:
            print("\n  Run A and Run B used different schedules, as intended.")
    return ok


def main():
    assert_single_variable()

    stages = build_stages()
    names = [s[0] for s in stages]

    ap = argparse.ArgumentParser(
        description="Run the noise schedule ablation end to end.",
        epilog="stages: " + ", ".join(names),
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--from", dest="start", choices=names,
                    help="resume from this stage")
    ap.add_argument("--only", nargs="+", choices=names,
                    help="run only these stages")
    ap.add_argument("--force", action="store_true",
                    help="redo stages whose output already exists")
    args = ap.parse_args()

    selected = stages
    if args.only:
        selected = [s for s in stages if s[0] in args.only]
    elif args.start:
        selected = stages[names.index(args.start):]

    problems = check_prerequisites([s[0] for s in selected])
    if problems and not args.dry_run:
        for p in problems:
            print(f"cannot start: {p}")
        return 1

    print(f"{len(selected)} stage(s): {', '.join(s[0] for s in selected)}")
    if args.dry_run:
        print("dry run - nothing will execute\n")

    started = time.perf_counter()
    skipped = []

    for name, description, commands, marker in selected:
        done = marker and (ROOT / marker).exists() and not args.force
        header = f"[{name}] {description}"

        if done and not args.dry_run:
            print(f"\n{header}\n  skipping, {marker} already exists "
                  f"(--force to redo)")
            skipped.append(name)
            continue

        print(f"\n{header}")
        stage_start = time.perf_counter()

        for cmd in commands:
            print(f"  $ {show(cmd)}")
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"\n[{name}] failed with exit code {result.returncode}")
                print("Nothing after this ran. Fix it, then resume with:")
                print(f"  python main.py --from {name}")
                return result.returncode

        if not args.dry_run:
            print(f"  [{name}] done in {time.perf_counter() - stage_start:.0f}s")

    if args.dry_run:
        return 0

    elapsed = time.perf_counter() - started
    print(f"\nFinished in {elapsed / 60:.0f} min")
    if skipped:
        print(f"Skipped (already present): {', '.join(skipped)}")

    if not verify_checksums():
        return 1

    print("\nNext: check outputs/runA/ckpt-2000 by eye before trusting "
          "report/metrics.csv.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
