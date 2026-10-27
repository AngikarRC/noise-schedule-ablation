# Noise Schedule Ablation in InstructPix2Pix

Group 38 — Foundation Models and Generative AI

Fine-tuning a pretrained image-to-image diffusion model on LLVIP under two
noise schedules, to measure what the power transform does to adaptation.

| | |
|---|---|
| Task | Visible-light image → infrared thermal image |
| Model | [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix) ([paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf)) |
| Dataset | [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) ([paper](https://arxiv.org/abs/2108.10831), ICCV 2021) |
| Parameterisation | θ_t = ᾱ_t (`scheduler.alphas_cumprod`) |
| Exponent | x = 2.35 |
| Seed | 20261026 |

Run A uses the native schedule, Run B uses ᾱ_t^2.35, and nothing else differs.

## Layout

```
main.py       runs every stage in order; prints each command first
baseline/     unmodified diffusers training example, kept for the diff
scripts/      our code; finetune.py is generated from the baseline
data/         split manifests (tracked) and the image cache (not tracked)
schedules/    per-run ᾱ arrays and SHA-256 checksums
logs/         env.json and per-run loss CSVs
outputs/      generated images and LoRA checkpoints
trajectories/ x_t dumps at 5 reverse-diffusion timesteps
report/       metrics, figures, code diff
```

## Setup

The RTX 5050 is Blackwell (sm_120). PyTorch wheels built against CUDA ≤ 12.6
install cleanly and then fail at the first GPU operation with *"no kernel
image is available for execution on the device"*. Install a CUDA 12.8+ build
first.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py
```

`check_env.py` verifies `sm_120` is in torch's arch list, runs a GPU smoke
test, and benchmarks the LoRA training step across resolutions and batch
sizes. It writes `logs/env.json`.

Three packages are deliberately **not** installed — `requirements.txt`
explains each: `datasets` (drags in pyarrow, which conflicts with the numpy
pin, and we load from files anyway), `bitsandbytes` (the Windows wheel ships
without its CUDA binary), `xformers` (can silently downgrade torch below
sm_120 support).

## Running it

Download LLVIP and unpack it to `LLVIP/` so that `LLVIP/visible/train` and
`LLVIP/infrared/train` exist, then:

```bash
python main.py
```

That runs all fourteen stages in order, about 3.5 hours. It prints every
command before executing it, skips stages whose output already exists, and
stops at the first failure with the command to resume from.

```bash
python main.py --dry-run          # print the commands, run nothing
python main.py --from train-a     # resume from a stage
python main.py --only eval        # one stage
python main.py --force            # redo stages that already have output
```

`main.py` is a convenience, not a dependency. Every stage below is a
standalone script with explicit arguments and can be run by hand; the
orchestrator just calls them in order. On startup it asserts that the Run A
and Run B commands differ only in `--run_mode`, `--output_dir` and
`--loss_csv`, and refuses to run if anything else has crept in.

## Reproducing stage by stage

The commands `main.py` issues, in order. Run these directly if you prefer.

### 1. Build the splits

```bash
python scripts/prepare_llvip.py --llvip LLVIP --out data \
  --resolution 512 --train-size 2000 --val-size 60 --test-size 100
```

LLVIP images are video frames and the first two filename digits are a scene
id. Frames 500 apart within a scene differ less than frames from different
scenes, so splits are cut on scene boundaries and frames are stride-sampled
inside each scene. A random split would put near-duplicates on both sides and
inflate test SSIM.

Scenes 01–16 train, 17/18/25 validate, 19–24 and 26 test. The split is
written once to `data/*.txt`; both runs read those files.

### 2. Verify the schedule

```bash
python scripts/schedule.py
```

Asserts 0 < ᾱ_t < 1 for all t in both schedules, prints the comparison table,
writes the full 1000-value arrays and a SHA-256 checksum per run, and saves
the log-SNR figure. Run this before training.

### 3. Generate the training script

```bash
python scripts/make_finetune.py --fetch
```

`--fetch` downloads the pinned baseline (overwriting any copy) and then
patches it. The download is done in Python rather than with `curl`, because
on PowerShell 5.1 `curl` is an alias for `Invoke-WebRequest` and rejects
`-L -o`. Omit `--fetch` to patch a baseline already on disk.

The URL points at the **v0.31.0 tag**, matching the pinned `diffusers`. The
copy on `main` imports symbols absent from 0.31.0 and fails at import;
`make_finetune.py` detects that and refuses to run.

`make_finetune.py` applies eleven anchored patches to the baseline and fails
loudly if any anchor does not match exactly once, so `scripts/finetune.py` is
reproducible rather than hand-edited. For the report:

```bash
diff -u baseline/train_instruct_pix2pix_ORIGINAL.py scripts/finetune.py > report/code_diff.txt
```

### 4. Train

Run A and Run B differ in `--run_mode` and `--output_dir`. Everything else is
identical.

```bash
python scripts/finetune.py --run_mode A --output_dir outputs/runA \
  --loss_csv logs/runA_loss.csv \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --llvip_data data --resolution 512 --center_crop \
  --train_batch_size 4 --gradient_accumulation_steps 1 \
  --max_train_steps 2500 --checkpointing_steps 500 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --mixed_precision fp16 --gradient_checkpointing \
  --lora_rank 16 --lora_alpha 16 --dataloader_num_workers 0 \
  --seed 20261026
```

```bash
python scripts/finetune.py --run_mode B --output_dir outputs/runB \
  --loss_csv logs/runB_loss.csv \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --llvip_data data --resolution 512 --center_crop \
  --train_batch_size 4 --gradient_accumulation_steps 1 \
  --max_train_steps 2500 --checkpointing_steps 500 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --mixed_precision fp16 --gradient_checkpointing \
  --lora_rank 16 --lora_alpha 16 --dataloader_num_workers 0 \
  --seed 20261026
```

**Why 2500 steps.** 2,000 training pairs at effective batch 4 is 500
optimizer steps per epoch, so 2,500 steps is exactly 5 epochs and a
checkpoint every 500 steps falls on an epoch boundary. The assignment asks
for samples at *5 equally spaced fine-tuning epochs*, so the checkpoints have
to coincide with epochs rather than being five arbitrary points in training.

Measured on the reference machine: 2.1 s/step at batch 4, 512px, so about
88 minutes per run. Note that `check_env.py` reports a much faster figure
because it feeds the UNet synthetic latents; real training also VAE-encodes
eight 512×512 images per step, which dominates.

### 4b. Per-epoch loss CSVs

```bash
python scripts/epoch_losses.py --in logs/runA_loss.csv --out logs/runA_epoch_loss.csv
python scripts/epoch_losses.py --in logs/runB_loss.csv --out logs/runB_epoch_loss.csv
```

`finetune.py` logs every optimizer step (`epoch,step,train_loss`), which is
more useful for plotting and for spotting instability. The submission wants
`epoch,train_loss`, so this collapses one into the other.

### 5. Run 0 and final inference

```bash
python scripts/infer.py --run_mode 0 --out outputs/run0 --seed 20261026

python scripts/infer.py --run_mode A --lora outputs/runA/checkpoint-2500 \
  --out outputs/runA/final --seed 20261026
python scripts/infer.py --run_mode B --lora outputs/runB/checkpoint-2500 \
  --out outputs/runB/final --seed 20261026
```

Run 0 takes no `--lora` — it is the pretrained checkpoint applied directly to
our test images, and `infer.py` rejects a `--lora` passed alongside
`--run_mode 0`.

### 6. Checkpoint samples

The same four validation images at each of the five checkpoints, both runs:

```bash
for run in A B; do
  for step in 500 1000 1500 2000 2500; do
    python scripts/infer.py --run_mode $run \
      --lora outputs/run$run/checkpoint-$step \
      --manifest val_fixed.txt \
      --out outputs/run$run/ckpt-$step \
      --seed 20261026
  done
done
```

On Windows PowerShell:

```powershell
foreach ($run in "A","B") {
  foreach ($step in 500,1000,1500,2000,2500) {
    python scripts\infer.py --run_mode $run `
      --lora "outputs\run$run\checkpoint-$step" `
      --manifest val_fixed.txt `
      --out "outputs\run$run\ckpt-$step" `
      --seed 20261026
  }
}
```

### 7. Trajectories and evaluation

```bash
python scripts/trajectory.py --run_mode A --lora outputs/runA/checkpoint-2500 \
  --out trajectories/runA --seed 20261026
python scripts/trajectory.py --run_mode B --lora outputs/runB/checkpoint-2500 \
  --out trajectories/runB --seed 20261026

python scripts/evaluate.py --runA outputs/runA/final --runB outputs/runB/final
```

## What makes this a single-variable ablation

- Both runs load the same pretrained checkpoint; Run B never starts from Run A.
- The split is computed once and written to disk; both runs read those files.
- The same seed goes to Python, NumPy, Torch and the dataloader generator.
- LoRA rank, alpha and target modules are identical. LoRA is used because full
  fine-tuning of the 0.9B UNet does not fit in 8 GB; the configuration being
  the same in both runs keeps the comparison clean.
- `schedule.py` writes a SHA-256 of the ᾱ array per run, and `finetune.py`,
  `infer.py` and `trajectory.py` each print the checksum of the schedule they
  actually used. Within a run, training and inference must match; between
  runs they must differ. Reference values: Run A `40873f7067e7f82d`,
  Run B `6fa947b412c6432e`.

## Notes

`finetune.py` is generated, not edited. To change it, edit the patch
definitions in `make_finetune.py` and regenerate — hand edits are lost on the
next run.

Checkpoints save only the LoRA adapter (~13 MB). Pass `--save_full_state` for
resumable training, which also writes the full UNet at ~3.3 GB per checkpoint.
