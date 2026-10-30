"""Saves the noisy sample x_t at five points of the reverse diffusion process.
I have decoded each capture through the VAE before writing it, since x_t is a
latent and would otherwise be four meaningless noise channels rather than a picture.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import build_pipeline, read_manifest
from llvip_data import PROMPT
from schedule import POWER_EXPONENT, schedule_checksum


def decode(pipe, latents):
    # At the first timestep Run B's x_t is a few hundred times larger than
    # anything the VAE was trained on, which overflows it and writes a black
    # frame, so I divide by its own spread before decoding and record that
    # divisor next to the image.
    z = torch.nan_to_num(latents.float(), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(z.std().item())
    if scale > 1.5:
        z = z / scale
    else:
        scale = 1.0
    was = next(pipe.vae.parameters()).dtype
    try:
        pipe.vae.float()
        with torch.no_grad():
            image = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
    finally:
        pipe.vae.to(was)
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_mode", required=True, choices=["A", "B"])
    ap.add_argument("--lora", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--manifest", default="test.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--captures", type=int, default=5)
    ap.add_argument("--image_guidance_scale", type=float, default=1.5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    args = ap.parse_args()

    pipe = build_pipeline(args.run_mode, args.lora)
    digest = schedule_checksum(pipe.scheduler)
    print(f"schedule checksum: {digest}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_manifest(args.data, args.manifest)[: args.samples]

    idx = [round(i * (args.steps - 1) / (args.captures - 1)) for i in range(args.captures)]
    capture_at = sorted(set(idx))
    print(f"capturing at sampler steps {capture_at} of {args.steps}")

    manifest = []

    for s, (name, path) in enumerate(items):
        stem = Path(name).stem
        source = Image.open(path).convert("RGB")
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed + s)
        captured = {}

        def callback(pipeline, step, timestep, kwargs):
            if step in capture_at:
                captured[step] = (int(timestep), kwargs["latents"].detach().clone())
            return kwargs

        pipe(
            PROMPT,
            image=source,
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        for step in capture_at:
            if step not in captured:
                print(f"  warning: step {step} not captured for {name}")
                continue
            t_value, latents = captured[step]
            img, display_scale = decode(pipe, latents[:1])
            fname = f"run{args.run_mode}_{stem}_step{step:02d}_t{t_value:04d}.png"
            img.save(out_dir / fname)
            manifest.append({
                "run": args.run_mode, "sample": name,
                "sampler_step": step, "timestep": t_value, "file": fname,
                "display_scale": round(display_scale, 2),
            })

        print(f"  {s+1}/{len(items)}  {name}")

    (out_dir / "_trajectory_info.json").write_text(json.dumps({
        "run_mode": args.run_mode,
        "schedule_checksum": digest,
        "exponent": POWER_EXPONENT if args.run_mode == "B" else 1.0,
        "num_inference_steps": args.steps,
        "capture_steps": capture_at,
        "seed": args.seed,
        "images": manifest,
    }, indent=2))

    print(f"wrote {len(manifest)} trajectory images to {out_dir}")

if __name__ == "__main__":
    main()
