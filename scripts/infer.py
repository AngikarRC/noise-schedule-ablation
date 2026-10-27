"""Generate infrared predictions from visible inputs.

Covers all three conditions:

    Run 0   no LoRA weights, native schedule  (out-of-distribution baseline)
    Run A   LoRA from run A, native schedule
    Run B   LoRA from run B, powered schedule

The schedule wiring matters more than it looks. A model fine-tuned under the
powered schedule and then sampled under the native one produces garbage, and
the failure looks exactly like "the modified schedule is bad". So --run_mode
drives both the weights and the sampler, and the schedule checksum is printed
on every call so the report can show training and inference agreed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedule import POWER_EXPONENT, apply_power_transform, schedule_checksum, verify_bounds
from llvip_data import PROMPT

MODEL_ID = "timbrooks/instruct-pix2pix"


def build_pipeline(run_mode, lora_path, device="cuda", dtype=torch.float16):
    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )

    if run_mode == "B":
        apply_power_transform(pipe.scheduler, POWER_EXPONENT)
    verify_bounds(pipe.scheduler, f"inference run {run_mode}")

    if lora_path:
        pipe.load_lora_weights(lora_path)
        print(f"loaded LoRA from {lora_path}")
    else:
        print("no LoRA weights (Run 0 baseline)")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def read_manifest(data_root, manifest):
    root = Path(data_root)
    rels = [l.strip() for l in (root / manifest).read_text().splitlines() if l.strip()]
    items = []
    for rel in rels:
        split, name = rel.split("/", 1)
        items.append((name, root / "cache" / split / "visible" / name))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_mode", required=True, choices=["0", "A", "B"])
    ap.add_argument("--lora", default=None, help="path to LoRA weights; omit for Run 0")
    ap.add_argument("--data", default="data")
    ap.add_argument("--manifest", default="test.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--image_guidance_scale", type=float, default=1.5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.run_mode == "0" and args.lora:
        raise SystemExit("Run 0 is the pretrained baseline; do not pass --lora")
    if args.run_mode in ("A", "B") and not args.lora:
        raise SystemExit(f"Run {args.run_mode} needs --lora")

    sched_mode = "B" if args.run_mode == "B" else "A"
    pipe = build_pipeline(sched_mode, args.lora)
    digest = schedule_checksum(pipe.scheduler)
    print(f"inference schedule checksum: {digest}")

    items = read_manifest(args.data, args.manifest)
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (name, path) in enumerate(items):
        # Reseed per image so results do not depend on position in the list.
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed + i)
        source = Image.open(path).convert("RGB")
        result = pipe(
            PROMPT,
            image=source,
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        result.save(out_dir / name)
        if (i + 1) % 10 == 0 or i == len(items) - 1:
            print(f"  {i+1}/{len(items)}")

    (out_dir / "_run_info.json").write_text(json.dumps({
        "run_mode": args.run_mode,
        "schedule_mode": sched_mode,
        "schedule_checksum": digest,
        "exponent": POWER_EXPONENT if sched_mode == "B" else 1.0,
        "lora": args.lora,
        "manifest": args.manifest,
        "seed": args.seed,
        "steps": args.steps,
        "image_guidance_scale": args.image_guidance_scale,
        "guidance_scale": args.guidance_scale,
        "count": len(items),
        "prompt": PROMPT,
    }, indent=2))

    print(f"wrote {len(items)} images to {out_dir}")


if __name__ == "__main__":
    main()
