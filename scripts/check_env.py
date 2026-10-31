"""Check GPU support and benchmark a representative LoRA training step.

This catches incompatible PyTorch/CUDA builds before the full training run.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def report_versions():
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"python   {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    info = {"python": sys.version.split()[0], "platform": platform.platform()}

    try:
        import torch
    except ImportError:
        print("torch    NOT INSTALLED")
        print("\nInstall with:")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return None, info

    print(f"torch    {torch.__version__}  (built for CUDA {torch.version.cuda})")
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda

    for name in ["diffusers", "transformers", "accelerate", "peft", "bitsandbytes"]:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{name:<9}{version}")
            info[name] = version
        except ImportError:
            print(f"{name:<9}NOT INSTALLED")
            info[name] = None

    return torch, info


def check_gpu(torch, info):
    print()
    print("=" * 60)
    print("GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Training will not run.")
        print("Check that you installed a CUDA build of torch, not the CPU wheel.")
        return False

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"device       {name}")
    print(f"capability   sm_{cap[0]}{cap[1]}")
    print(f"memory       {total:.1f} GiB")

    info["gpu"] = name
    info["capability"] = f"sm_{cap[0]}{cap[1]}"
    info["vram_gib"] = round(total, 2)

    supported = torch.cuda.get_arch_list()
    print(f"torch archs  {' '.join(supported)}")

    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in supported:
        print()
        print(f"PROBLEM: this torch build has no kernels for {arch}.")
        print("It will fail the moment a tensor touches the GPU.")
        print("Reinstall with a CUDA 12.8+ build:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return False

    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = a @ a
        torch.cuda.synchronize()
        print(f"smoke test   passed (result mean {b.float().mean().item():.4f})")
    except Exception as exc:
        print(f"smoke test   FAILED: {exc}")
        return False

    free, _ = torch.cuda.mem_get_info()
    print(f"free now     {free / 1024**3:.1f} GiB")
    if free / 1024**3 < 6.0:
        print("             (low — close other GPU applications before training)")

    return True


def time_batch(torch, unet, batch_size, steps, lora_params, latent_size):
    latents = torch.randn(batch_size, 8, latent_size, latent_size,
                          device="cuda", dtype=torch.float16)
    encoder_states = torch.randn(batch_size, 77, 768, device="cuda", dtype=torch.float16)
    timesteps = torch.randint(0, 1000, (batch_size,), device="cuda")

    opt = torch.optim.AdamW(lora_params, lr=1e-4)

    def one():
        loss = unet(latents, timesteps, encoder_states).sample.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        one()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    del opt, latents, encoder_states, timesteps
    torch.cuda.empty_cache()

    return elapsed / steps, torch.cuda.max_memory_allocated() / 1024**3


def benchmark(torch, info, steps, resolutions):
    print()
    print("=" * 60)
    print(f"STEP TIME ({steps} micro-batches each)")
    print("=" * 60)

    try:
        from diffusers import UNet2DConditionModel
        from peft import LoraConfig
    except ImportError as exc:
        print(f"missing dependency ({exc.name}), skipping")
        return

    print("loading UNet from timbrooks/instruct-pix2pix ...")
    unet = UNet2DConditionModel.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="unet", torch_dtype=torch.float16
    ).to("cuda")
    unet.enable_gradient_checkpointing()

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=16, lora_alpha=16, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    unet.train()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    total = sum(p.numel() for p in unet.parameters())
    trainable = sum(p.numel() for p in lora_params)
    print(f"LoRA rank 16: {trainable:,} trainable of {total:,} "
          f"({100 * trainable / total:.2f}%)")
    print()

    results = {}
    print(f"{'res':>6}{'batch':>7}{'s/micro-batch':>16}{'img/s':>10}{'peak GiB':>11}")
    for res in resolutions:
        latent = res // 8
        for bs in [1, 2, 4, 8]:
            try:
                per, peak = time_batch(torch, unet, bs, steps, lora_params, latent)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{res:>6}{bs:>7}{'out of memory':>16}")
                break
            results[(res, bs)] = (per, peak)
            print(f"{res:>6}{bs:>7}{per:>16.3f}{bs / per:>10.2f}{peak:>11.2f}")

    info["bench"] = {f"{r}px_b{b}": {"sec": round(v[0], 4), "peak_gib": round(v[1], 2)}
                     for (r, b), v in results.items()}

    print()
    print("Wall-clock for ONE run at effective batch 4.")
    print("An optimizer step costs `accum` micro-batches, so 2500 steps at")
    print("accum 4 is 10,000 forward+backward passes, not 2,500.")
    print()
    print(f"{'res':>6}{'batch':>7}{'accum':>7}{'2000':>10}{'3000':>10}{'5000':>10}")
    for (res, bs), (per, _) in results.items():
        if bs > 4:
            continue
        accum = max(1, 4 // bs)
        cols = "".join(f"{n * accum * per / 60:>9.1f}m" for n in (2000, 3000, 5000))
        print(f"{res:>6}{bs:>7}{accum:>7}{cols}")

    for res in resolutions:
        at_res = {b: v for (r, b), v in results.items() if r == res}
        if at_res:
            best = min(at_res, key=lambda b: at_res[b][0] / b)
            print(f"\n{res}px: fastest per image at batch {best} "
                  f"({at_res[best][0] / best:.3f} s/image, "
                  f"{at_res[best][1]:.2f} GiB peak)")


def main():
    ap = argparse.ArgumentParser(description="Check the GPU setup and benchmark training.")
    ap.add_argument("--steps", type=int, default=20, help="Benchmark steps per resolution.")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[256, 512],
                    help="Image sizes to benchmark.")
    ap.add_argument("--skip-benchmark", action="store_true", help="Skip the training benchmark.")
    ap.add_argument("--out", type=Path, default=Path("logs/env.json"), help="Report file path.")
    args = ap.parse_args()

    torch, info = report_versions()
    if torch is None:
        sys.exit(1)

    if not check_gpu(torch, info):
        sys.exit(1)

    if not args.skip_benchmark:
        benchmark(torch, info, args.steps, args.resolutions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
