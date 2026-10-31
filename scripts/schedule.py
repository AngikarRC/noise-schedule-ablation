"""Build and verify the native and powered alpha_bar schedules.

Checks their bounds, records checksums, and saves a log-SNR comparison figure.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

POWER_EXPONENT = 2.35  # ADDED - declared exponent, Phase 1 item 6

MODEL_ID = "timbrooks/instruct-pix2pix"


def _as_numpy(x):
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def transformed_alphas_cumprod(alphas_cumprod, exponent):
    abar = _as_numpy(alphas_cumprod)
    abar_x = abar ** float(exponent)

    alphas = np.empty_like(abar_x)
    alphas[0] = abar_x[0]
    alphas[1:] = abar_x[1:] / abar_x[:-1]
    betas = 1.0 - alphas

    return abar_x, alphas, betas


def apply_power_transform(scheduler, exponent=POWER_EXPONENT):
    if exponent == 1.0:
        return scheduler

    dtype = scheduler.alphas_cumprod.dtype
    device = scheduler.alphas_cumprod.device

    abar_x, alphas, betas = transformed_alphas_cumprod(scheduler.alphas_cumprod, exponent)

    if not (betas > 0).all() or not (betas < 1).all():
        raise ValueError(
            f"transform produced betas outside (0,1): "
            f"min={betas.min():.6e} max={betas.max():.6e}"
        )

    scheduler.alphas_cumprod = torch.tensor(abar_x, dtype=dtype, device=device)
    scheduler.alphas = torch.tensor(alphas, dtype=dtype, device=device)
    scheduler.betas = torch.tensor(betas, dtype=dtype, device=device)

    if hasattr(scheduler, "final_alpha_cumprod"):
        set_to_one = getattr(scheduler.config, "set_alpha_to_one", True)
        value = 1.0 if set_to_one else abar_x[0]
        scheduler.final_alpha_cumprod = torch.tensor(value, dtype=dtype, device=device)

    return scheduler


def schedule_checksum(scheduler):
    abar = _as_numpy(scheduler.alphas_cumprod)
    return hashlib.sha256(abar.tobytes()).hexdigest()[:16]


def schedule_stats(scheduler):
    abar = _as_numpy(scheduler.alphas_cumprod)
    snr = abar / (1.0 - abar)
    T = len(abar)
    below_half = np.argmax(abar < 0.5) if (abar < 0.5).any() else None

    return {
        "T": T,
        "abar_first": float(abar[0]),
        "abar_mid": float(abar[T // 2 - 1]),
        "abar_last": float(abar[-1]),
        "signal_amplitude_last": float(np.sqrt(abar[-1])),
        "snr_last": float(snr[-1]),
        "log10_snr_first": float(np.log10(snr[0])),
        "log10_snr_last": float(np.log10(snr[-1])),
        "crosses_half_at": int(below_half) + 1 if below_half is not None else None,
        "min": float(abar.min()),
        "max": float(abar.max()),
    }


def verify_bounds(scheduler, label):
    abar = _as_numpy(scheduler.alphas_cumprod)
    ok_low = bool((abar > 0.0).all())
    ok_high = bool((abar < 1.0).all())
    print(f"[{label}] 0 < abar_t     : {ok_low}   (min {abar.min():.6e})")
    print(f"[{label}] abar_t < 1     : {ok_high}  (max {abar.max():.6f})")
    assert ok_low and ok_high, f"{label}: alpha_bar left the open interval (0,1)"
    return True


def load_scheduler(run_mode, model_id=MODEL_ID, subfolder="scheduler"):
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder=subfolder)
    if run_mode == "B":
        apply_power_transform(scheduler, POWER_EXPONENT)
    elif run_mode != "A":
        raise ValueError(f"run_mode must be 'A' or 'B', got {run_mode!r}")
    return scheduler


def log_schedule(scheduler, run_mode, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abar = _as_numpy(scheduler.alphas_cumprod)
    alphas = _as_numpy(scheduler.alphas)
    betas = _as_numpy(scheduler.betas)
    snr = abar / (1.0 - abar)

    csv_path = out_dir / f"schedule_run{run_mode}.csv"
    with csv_path.open("w") as fh:
        fh.write("t,beta,alpha,alpha_bar,snr,log10_snr\n")
        for t in range(len(abar)):
            fh.write(f"{t+1},{betas[t]:.12e},{alphas[t]:.12e},"
                     f"{abar[t]:.12e},{snr[t]:.12e},{np.log10(snr[t]):.12e}\n")

    digest = schedule_checksum(scheduler)
    stats = schedule_stats(scheduler)
    stats["checksum"] = digest
    stats["run_mode"] = run_mode
    stats["exponent"] = POWER_EXPONENT if run_mode == "B" else 1.0

    (out_dir / f"schedule_run{run_mode}.json").write_text(json.dumps(stats, indent=2))

    print(f"[run {run_mode}] checksum {digest}  ->  {csv_path.name}")
    return digest


def plot_comparison(sched_a, sched_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abar_a = _as_numpy(sched_a.alphas_cumprod)
    abar_b = _as_numpy(sched_b.alphas_cumprod)
    t = np.arange(1, len(abar_a) + 1)

    snr_a = abar_a / (1 - abar_a)
    snr_b = abar_b / (1 - abar_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(t, abar_a, label=r"Run A:  $\bar{\alpha}_t$", lw=1.8)
    ax1.plot(t, abar_b, label=rf"Run B:  $\bar{{\alpha}}_t^{{{POWER_EXPONENT}}}$", lw=1.8)
    ax1.axhline(0.5, color="grey", ls=":", lw=1)
    ax1.set_xlabel("diffusion timestep $t$")
    ax1.set_ylabel(r"$\bar{\alpha}_t$")
    ax1.set_title("Cumulative signal retention")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(t, np.log10(snr_a), label="Run A", lw=1.8)
    ax2.plot(t, np.log10(snr_b), label="Run B", lw=1.8)
    ax2.set_xlabel("diffusion timestep $t$")
    ax2.set_ylabel(r"$\log_{10}$ SNR")
    ax2.set_title("Signal-to-noise ratio")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare and export both noise schedules.")
    ap.add_argument("--model", default=MODEL_ID, help="Pretrained model ID or path.")
    ap.add_argument("--out-dir", type=Path, default=Path("schedules"), help="Schedule output folder.")
    ap.add_argument("--figure", type=Path, default=Path("report/figures/fig1_schedule.png"),
                    help="Log-SNR figure path.")
    args = ap.parse_args()

    print(f"exponent x = {POWER_EXPONENT}\n")

    sched_a = load_scheduler("A", args.model)
    sched_b = load_scheduler("B", args.model)

    print("--- bounds (Phase 1 item 5) ---")
    verify_bounds(sched_a, "A")
    verify_bounds(sched_b, "B")

    print("\n--- schedule summary ---")
    stats_a, stats_b = schedule_stats(sched_a), schedule_stats(sched_b)
    rows = [
        ("abar at t=1", "abar_first", "{:.6f}"),
        ("abar at t=T/2", "abar_mid", "{:.6f}"),
        ("abar at t=T", "abar_last", "{:.6e}"),
        ("signal amplitude at T", "signal_amplitude_last", "{:.6f}"),
        ("log10 SNR at t=1", "log10_snr_first", "{:+.3f}"),
        ("log10 SNR at t=T", "log10_snr_last", "{:+.3f}"),
    ]
    print(f"{'quantity':<24}{'Run A':>16}{'Run B':>16}")
    for label, key, fmt in rows:
        print(f"{label:<24}{fmt.format(stats_a[key]):>16}{fmt.format(stats_b[key]):>16}")
    print(f"{'abar crosses 0.5 at t':<24}"
          f"{str(stats_a['crosses_half_at']):>16}{str(stats_b['crosses_half_at']):>16}")

    print("\n--- checksums ---")
    digest_a = log_schedule(sched_a, "A", args.out_dir)
    digest_b = log_schedule(sched_b, "B", args.out_dir)
    assert digest_a != digest_b, "A and B produced identical schedules"

    (args.out_dir / "checksums.txt").write_text(
        f"runA {digest_a}\nrunB {digest_b}\nexponent {POWER_EXPONENT}\n"
    )

    print()
    plot_comparison(sched_a, sched_b, args.figure)

    print("\nVerification passed. Paste the table above into the report's "
          "Schedule Analysis section.")

if __name__ == "__main__":
    main()
