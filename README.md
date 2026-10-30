# Noise Schedule Ablation in InstructPix2Pix

Group 38, Foundation Models and Generative AI.

This repository fine-tunes a pretrained image-to-image diffusion model on LLVIP twice. The two runs are identical apart from the noise schedule: one uses the schedule the model shipped with, the other raises it to the power 2.35. The point is to measure what that change does to adaptation.

The task is visible-light to infrared translation. Given an ordinary photograph of a night street scene, the model produces the thermal image a heat camera would have recorded of the same scene.

| | |
|---|---|
| Model | timbrooks/instruct-pix2pix |
| Dataset | LLVIP, 2160 of 15488 pairs, at 512x512 |
| Noise parameter | alpha_bar, the cumulative signal retention term |
| Exponent | 2.35 |
| Seed | 20261026 |
| Training | 5 epochs, 2500 steps, LoRA rank 16 |

There are three runs. Run 0 applies the pretrained model with no fine-tuning and acts as the baseline. Run A fine-tunes with the original schedule. Run B fine-tunes with the powered schedule. Nothing else differs between A and B.

## Setting up

The RTX 5050 is a Blackwell card. PyTorch wheels built against CUDA 12.6 or earlier install without complaining and then fail at the first GPU operation with "no kernel image is available for execution on the device". Install a CUDA 12.8 build first.

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    python scripts\check_env.py

check_env.py confirms sm_120 is in torch's architecture list, runs a GPU smoke test and benchmarks a LoRA training step. It writes logs/env.json.

Three packages are deliberately left out. The datasets library pulls in pyarrow, which demands a different numpy than the one pinned here, and we read LLVIP from files anyway. The bitsandbytes wheel for Windows ships without its CUDA binary, and with LoRA the optimizer states are small enough that 8-bit Adam buys nothing. xformers can quietly downgrade torch to a build without sm_120 support.

## Running the experiment

Download LLVIP and unpack it so that LLVIP/visible/train and LLVIP/infrared/train exist, then run:

    python main.py

That executes all fifteen stages in order and takes roughly four hours. It prints every command before running it, skips any stage whose output already exists, and stops at the first failure with the command needed to resume.

    python main.py --dry-run          print the commands without running them
    python main.py --from train-b     resume from a stage
    python main.py --only eval        run one stage
    python main.py --force            redo stages that already have output

main.py is a convenience. Every stage below is a standalone script and can be run by hand. On startup main.py checks that the Run A and Run B commands differ only in run_mode, output_dir and loss_csv, and refuses to run if anything else has crept in.

## The stages, one at a time

### Prepare the data

    python scripts\prepare_llvip.py --llvip LLVIP --out data ^
      --resolution 512 --train-size 2000 --val-size 60 --test-size 100

LLVIP images are video frames and the first two digits of a filename identify the capture session. Frames from the same session are far more alike than frames from different sessions, so a random split would put near-identical images on both sides of the train/test boundary and inflate the test scores through memorisation. Every split boundary therefore falls on a session boundary, and frames are sampled with a stride within each session. Sessions 01 to 16 are used for training, 17, 18 and 25 for validation, and LLVIP's own test sessions 19 to 24 and 26 for testing.

The split is written once to data/train.txt, data/val.txt and data/test.txt. Both training runs read those files rather than recomputing the split.

### Verify the schedule

    python scripts\schedule.py

This asserts that alpha_bar stays strictly between 0 and 1 at every timestep for both schedules, prints the comparison table, writes all 1000 values per run to CSV, records a SHA-256 of each schedule, and saves the log-SNR figure to report/figures. Run it before training, because it takes seconds and the experiment is pointless if the transform is wrong.

The two checksums are 40873f7067e7f82d for Run A and 6fa947b412c6432e for Run B.

### Generate the training script

    python scripts\make_finetune.py --fetch

The baseline is Hugging Face's InstructPix2Pix training example at tag v0.31.0, which matches the pinned diffusers version. The copy on main imports symbols that do not exist in 0.31.0 and fails at import time; make_finetune.py detects that and refuses to run.

scripts/finetune.py is generated, not hand-written. make_finetune.py applies fifteen patches to the untouched baseline and fails if any anchor does not match exactly once, so the result is reproducible rather than something that drifted. Editing finetune.py directly has no effect, because the next regeneration overwrites it. To change training behaviour, edit the patch definitions at the top of make_finetune.py.

The --fetch flag downloads the baseline in Python rather than shelling out to curl, because on PowerShell 5.1 curl is an alias for Invoke-WebRequest and rejects the -L and -o flags.

For the report, produce the side-by-side diff with:

    diff -u baseline\train_instruct_pix2pix_ORIGINAL.py scripts\finetune.py > report\code_diff.txt

### Train Run A

    python scripts\finetune.py --run_mode A --output_dir outputs\runA ^
      --loss_csv logs\runA_loss.csv ^
      --pretrained_model_name_or_path timbrooks/instruct-pix2pix ^
      --llvip_data data --resolution 512 --center_crop ^
      --train_batch_size 4 --gradient_accumulation_steps 1 ^
      --max_train_steps 2500 --checkpointing_steps 500 ^
      --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 ^
      --mixed_precision fp16 --gradient_checkpointing ^
      --lora_rank 16 --lora_alpha 16 --dataloader_num_workers 0 ^
      --seed 20261026

### Train Run B

Identical except for run_mode, output_dir and loss_csv:

    python scripts\finetune.py --run_mode B --output_dir outputs\runB ^
      --loss_csv logs\runB_loss.csv ^
      --pretrained_model_name_or_path timbrooks/instruct-pix2pix ^
      --llvip_data data --resolution 512 --center_crop ^
      --train_batch_size 4 --gradient_accumulation_steps 1 ^
      --max_train_steps 2500 --checkpointing_steps 500 ^
      --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 ^
      --mixed_precision fp16 --gradient_checkpointing ^
      --lora_rank 16 --lora_alpha 16 --dataloader_num_workers 0 ^
      --seed 20261026

2000 training pairs at an effective batch of 4 is 500 optimizer steps per epoch, so 2500 steps is exactly 5 epochs and a checkpoint every 500 steps lands on an epoch boundary. The assignment asks for samples at five equally spaced epochs, which is why the step count is tied to the dataset size this way.

On the reference machine each run takes about 88 minutes at 2.1 seconds per step. check_env.py reports a much faster figure because it feeds the UNet synthetic latents; real training also encodes eight 512x512 images through the VAE on every step, and that dominates.

In the first minute of each run, check the schedule checksum printed in the log. Run A must print 40873f7067e7f82d and Run B must print 6fa947b412c6432e. If both print the same value the transform is not being applied and everything downstream is worthless.

### Per-epoch loss files

    python scripts\epoch_losses.py --in logs\runA_loss.csv --out logs\runA_epoch_loss.csv
    python scripts\epoch_losses.py --in logs\runB_loss.csv --out logs\runB_epoch_loss.csv

finetune.py logs every optimizer step, which is more useful for plotting and for spotting instability. The submission asks for epoch and train_loss, so this collapses one into the other.

### Generate images

    python scripts\infer.py --run_mode 0 --out outputs\run0 --seed 20261026

    python scripts\infer.py --run_mode A --lora outputs\runA\checkpoint-2500 ^
      --out outputs\runA\final --seed 20261026

    python scripts\infer.py --run_mode B --lora outputs\runB\checkpoint-2500 ^
      --out outputs\runB\final --seed 20261026

Run 0 takes no LoRA weights, and infer.py rejects a --lora passed alongside --run_mode 0.

run_mode drives both the weights and the sampler's schedule. A model fine-tuned under the powered schedule and then sampled under the native one produces garbage, and the failure looks exactly like the modified schedule being at fault, so one flag controls both.

### Samples at each epoch boundary

    foreach ($run in "A","B") {
      foreach ($s in 500,1000,1500,2000,2500) {
        python scripts\infer.py --run_mode $run `
          --lora "outputs\run$run\checkpoint-$s" `
          --manifest val_fixed.txt `
          --out "outputs\run$run\ckpt-$s" `
          --seed 20261026
      }
    }

The same four validation images with the same seed at every checkpoint, so differences between folders are learning rather than sampling noise.

### Inference trajectories

    python scripts\trajectory.py --run_mode A --lora outputs\runA\checkpoint-2500 ^
      --out trajectories\runA --seed 20261026

    python scripts\trajectory.py --run_mode B --lora outputs\runB\checkpoint-2500 ^
      --out trajectories\runB --seed 20261026

This saves the noisy sample x_t at five points of the reverse process for three images. x_t is a latent, not an image, so each capture is decoded through the VAE before being written out.

### Evaluate

    python scripts\evaluate.py --runA outputs\runA\final --runB outputs\runB\final

Reports SSIM and PSNR, plus luminance error and contrast ratio. The last two exist because the hypothesis concerns global brightness accuracy, and SSIM is too local a measure to show that reliably.

## What keeps this a single-variable comparison

Both runs load the same pretrained checkpoint, and Run B never starts from Run A. The split is computed once and written to disk, so both runs read the same files. The same seed is given to Python, numpy, torch and the dataloader generator. The LoRA rank, alpha and target modules are identical.

LoRA is used because full fine-tuning of the 862 million parameter UNet does not fit in 8 GB of VRAM. The configuration being the same in both runs keeps the comparison clean.

Every stage that touches noise writes a SHA-256 of the alpha_bar array it used. schedule.py records the expected values before training. finetune.py records what training actually used. infer.py and trajectory.py record what inference used. Within a run the training and inference values must match, and between runs they must differ. That is the evidence that Run B used the modified schedule.

## Layout

    main.py          runs every stage in order
    baseline/        the untouched upstream training script
    scripts/         our code; finetune.py is generated from the baseline
    data/            split manifests (tracked) and the image cache (not tracked)
    schedules/       per-run alpha_bar arrays and checksums
    logs/            env.json and the loss CSVs
    outputs/         generated images and LoRA checkpoints
    trajectories/    the x_t captures
    report/          metrics, figures and the code diff
