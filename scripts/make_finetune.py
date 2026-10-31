"""Generate scripts/finetune.py from the pinned upstream training example.

Checked patches keep the baseline untouched and make the generated changes repeatable.
"""

import argparse
import sys
from pathlib import Path

BASELINE = Path("baseline/train_instruct_pix2pix_ORIGINAL.py")
TARGET = Path("scripts/finetune.py")

DIFFUSERS_VERSION = "v0.31.0"

BASELINE_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_VERSION}"
    "/examples/instruct_pix2pix/train_instruct_pix2pix.py"
)

HEADER = f'''# ---------------------------------------------------------------------------
# Generated from huggingface/diffusers {DIFFUSERS_VERSION}, example
#   examples/instruct_pix2pix/train_instruct_pix2pix.py
# Source copy: baseline/train_instruct_pix2pix_ORIGINAL.py
# Project changes are marked ADDED or MODIFIED. Regenerate with:
#   python scripts/make_finetune.py
# ---------------------------------------------------------------------------
'''

IMPORTS = '''# MODIFIED - Use the local LLVIP loader to avoid the datasets pyarrow dependency.

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ADDED
from schedule import (  # ADDED
    POWER_EXPONENT,  # ADDED
    apply_power_transform,  # ADDED
    log_schedule,  # ADDED
    schedule_checksum,  # ADDED
    verify_bounds,  # ADDED
)  # ADDED
from llvip_data import LLVIPPairs, collate_fn as llvip_collate_fn, PROMPT  # ADDED
'''

NEW_ARGS = '''    parser.add_argument(  # ADDED
        "--run_mode",  # ADDED
        type=str,  # ADDED
        required=True,  # ADDED
        choices=["A", "B"],  # ADDED
        help="A = native; B = powered schedule",  # ADDED
    )  # ADDED
    parser.add_argument(  # ADDED
        "--llvip_data",  # ADDED
        type=str,  # ADDED
        default="data",  # ADDED
        help="Prepared LLVIP data folder.",  # ADDED
    )  # ADDED
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")  # ADDED
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")  # ADDED
    parser.add_argument(  # ADDED
        "--save_full_state",  # ADDED
        action="store_true",  # ADDED
        help="Save a resumable training checkpoint.",  # ADDED
    )  # ADDED
    parser.add_argument(  # ADDED
        "--loss_csv",  # ADDED
        type=str,  # ADDED
        default=None,  # ADDED
        help="Per-step loss CSV path.",  # ADDED
    )  # ADDED
    parser.add_argument(  # ADDED
        "--dataset_name",
'''

SCHEDULE_HOOK = '''    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    verify_bounds(noise_scheduler, f"run {args.run_mode} (native)")  # ADDED
    if args.run_mode == "B":  # ADDED
        apply_power_transform(noise_scheduler, POWER_EXPONENT)  # ADDED
        verify_bounds(noise_scheduler, f"run {args.run_mode} (powered)")  # ADDED
    _schedule_digest = log_schedule(noise_scheduler, args.run_mode, Path(args.output_dir) / "schedule")  # ADDED
    logger.info(f"run {args.run_mode}: schedule checksum {_schedule_digest}")  # ADDED
'''

LORA = '''        unet.enable_gradient_checkpointing()

    # ADDED - Full fine-tuning exceeds 8 GB of VRAM. Keep LoRA settings the same
    # across both runs so the noise schedule is the only training difference.
    from peft import LoraConfig  # ADDED

    unet.requires_grad_(False)  # ADDED
    lora_config = LoraConfig(  # ADDED
        r=args.lora_rank,  # ADDED
        lora_alpha=args.lora_alpha,  # ADDED
        init_lora_weights="gaussian",  # ADDED
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # ADDED
    )  # ADDED
    unet.add_adapter(lora_config)  # ADDED
    lora_params = [p for p in unet.parameters() if p.requires_grad]  # ADDED
    logger.info(  # ADDED
        f"LoRA rank {args.lora_rank}: {sum(p.numel() for p in lora_params):,} "  # ADDED
        f"trainable of {sum(p.numel() for p in unet.parameters()):,}"  # ADDED
    )  # ADDED
'''

OPTIMIZER = '''    optimizer = optimizer_cls(
        lora_params,  # MODIFIED - was unet.parameters(); only the adapter trains
'''

CONV_IN = '''    # MODIFIED - InstructPix2Pix already has eight input channels; skip expansion.
    if unet.conv_in.in_channels == 8:  # ADDED
        logger.info("UNet already has 8 input channels; keeping conv_in as is.")  # ADDED
    else:  # ADDED
        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
'''

DATASET = '''    # MODIFIED - Load LLVIP pairs from the manifests and cache built by prepare_llvip.py.
    train_dataset = LLVIPPairs(  # ADDED
        data_root=args.llvip_data,  # ADDED
        manifest="train.txt",  # ADDED
        tokenizer=tokenizer,  # ADDED
        prompt=PROMPT,  # ADDED
        random_flip=args.random_flip,  # ADDED
    )  # ADDED
    logger.info(f"LLVIP train pairs: {len(train_dataset)}")  # ADDED

    train_dataloader = torch.utils.data.DataLoader(  # ADDED
        train_dataset,  # ADDED
        shuffle=True,  # ADDED
        collate_fn=llvip_collate_fn,  # ADDED
        batch_size=args.train_batch_size,  # ADDED
        num_workers=args.dataloader_num_workers,  # ADDED
        generator=torch.Generator().manual_seed(args.seed),  # ADDED
    )  # ADDED
'''

SAVE_LORA = '''                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # MODIFIED - Save adapter weights for inference. Use
                        # --save_full_state only when a resumable state is needed.
                        from peft.utils import get_peft_model_state_dict  # ADDED

                        if args.save_full_state:  # ADDED
                            accelerator.save_state(save_path)  # ADDED
                        StableDiffusionInstructPix2PixPipeline.save_lora_weights(  # ADDED
                            save_directory=save_path,  # ADDED
                            unet_lora_layers=get_peft_model_state_dict(unwrap_model(unet)),  # ADDED
                            safe_serialization=True,  # ADDED
                        )  # ADDED
'''

FINAL_SAVE = '''        # MODIFIED - Save the fine-tuned adapter; the pretrained pipeline is shared.
        from peft.utils import get_peft_model_state_dict  # ADDED

        StableDiffusionInstructPix2PixPipeline.save_lora_weights(  # ADDED
            save_directory=args.output_dir,  # ADDED
            unet_lora_layers=get_peft_model_state_dict(unwrap_model(unet)),  # ADDED
            safe_serialization=True,  # ADDED
        )  # ADDED
'''

LOSS_LOG = '''                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.loss_csv and accelerator.is_main_process:  # ADDED
                    _csv = Path(args.loss_csv)  # ADDED
                    _csv.parent.mkdir(parents=True, exist_ok=True)  # ADDED
                    if not _csv.exists():  # ADDED
                        _csv.write_text("epoch,step,train_loss\\n")  # ADDED
                    with _csv.open("a") as _fh:  # ADDED
                        _fh.write(f"{epoch},{global_step},{train_loss:.6f}\\n")  # ADDED
'''


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"anchor '{label}' matched {count} times, expected 1")
    return text.replace(old, new)


def cut_block(text, start_marker, end_marker, replacement, label):
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"start anchor for '{label}' not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"end anchor for '{label}' not found")
    return text[:start] + replacement + text[end:]


def fetch_baseline():
    from urllib.request import urlopen

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {DIFFUSERS_VERSION} baseline ...")
    with urlopen(BASELINE_URL, timeout=60) as response:
        if response.status != 200:
            raise SystemExit(f"download failed: HTTP {response.status}")
        body = response.read().decode("utf-8")

    if "def parse_args" not in body:
        raise SystemExit("downloaded file does not look like the training example")

    BASELINE.write_text(body, encoding="utf-8", newline="\n")
    print(f"  wrote {BASELINE} ({len(body.splitlines())} lines)")


def main():
    ap = argparse.ArgumentParser(description="Generate the project fine-tuning script.")
    ap.add_argument("--fetch", action="store_true",
                    help="Download the pinned baseline first.")
    args = ap.parse_args()

    if args.fetch or not BASELINE.exists():
        fetch_baseline()

    src = BASELINE.read_text(encoding="utf-8")

    if "DIFFUSERS_REQUEST_TIMEOUT" in src:
        raise SystemExit(
            f"{BASELINE} looks like the version from main, not {DIFFUSERS_VERSION}.\n"
            "It imports symbols that are absent from the pinned diffusers release.\n"
            "Re-fetch it:\n  python scripts/make_finetune.py --fetch"
        )

    src = replace_once(src, "from datasets import load_dataset\n", IMPORTS, "imports")
    src = replace_once(
        src, "import datasets\n",
        "# MODIFIED - LLVIP is loaded from local files; datasets is not required.\n",
        "drop datasets import",
    )
    src = replace_once(
        src,
        "        datasets.utils.logging.set_verbosity_warning()\n",
        "        # MODIFIED - Remove unused datasets logging.\n",
        "verbosity warning",
    )
    src = replace_once(
        src,
        "        datasets.utils.logging.set_verbosity_error()\n",
        "        # MODIFIED - Remove unused datasets logging.\n",
        "verbosity error",
    )
    src = replace_once(src, '    parser.add_argument(\n        "--dataset_name",\n',
                       NEW_ARGS, "new args")
    src = replace_once(
        src,
        '"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""',
        '"""Fine-tune InstructPix2Pix on LLVIP with LoRA and the selected noise schedule."""',
        "training module description",
    )
    src = replace_once(
        src,
        'parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")',
        'parser = argparse.ArgumentParser(description="Fine-tune InstructPix2Pix with LoRA.")',
        "training CLI description",
    )
    src = replace_once(
        src,
        '    # Load scheduler, tokenizer and models.\n'
        '    noise_scheduler = DDPMScheduler.from_pretrained('
        'args.pretrained_model_name_or_path, subfolder="scheduler")\n',
        SCHEDULE_HOOK,
        "schedule hook",
    )
    src = replace_once(
        src,
        "    # Sanity checks\n"
        "    if args.dataset_name is None and args.train_data_dir is None:\n"
        '        raise ValueError("Need either a dataset name or a training folder.")\n',
        "    # MODIFIED - Check the LLVIP manifest instead of upstream dataset args.\n"
        "    if not (Path(args.llvip_data) / 'train.txt').exists():  # ADDED\n"
        "        raise ValueError(  # ADDED\n"
        "            f\"{args.llvip_data}/train.txt not found - run prepare_llvip.py first\"  # ADDED\n"
        "        )  # ADDED\n",
        "dataset sanity check",
    )
    src = replace_once(
        src,
        '    parser.add_argument("--seed", type=int, default=None, '
        'help="A seed for reproducible training.")\n',
        '    parser.add_argument(  # MODIFIED - require the same seed for both runs\n'
        '        "--seed", type=int, required=True,  # to compare schedules fairly\n'
        '        help="A seed for reproducible training.",\n'
        '    )\n',
        "seed required",
    )
    src = replace_once(
        src,
        '    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")\n'
        "    in_channels = 8\n"
        "    out_channels = unet.conv_in.out_channels\n"
        "    unet.register_to_config(in_channels=in_channels)\n"
        "\n"
        "    with torch.no_grad():\n"
        "        new_conv_in = nn.Conv2d(\n"
        "            in_channels, out_channels, unet.conv_in.kernel_size, "
        "unet.conv_in.stride, unet.conv_in.padding\n"
        "        )\n"
        "        new_conv_in.weight.zero_()\n"
        "        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)\n"
        "        unet.conv_in = new_conv_in\n",
        CONV_IN,
        "conv_in widening",
    )
    src = replace_once(src, "        unet.enable_gradient_checkpointing()\n", LORA, "lora")
    src = replace_once(src, "    optimizer = optimizer_cls(\n        unet.parameters(),\n",
                       OPTIMIZER, "optimizer")
    src = cut_block(
        src,
        "    # In distributed training, the load_dataset function guarantees",
        "    # Scheduler and math around the number of training steps.",
        DATASET,
        "dataset block",
    )
    src = replace_once(
        src,
        '                accelerator.log({"train_loss": train_loss}, step=global_step)\n',
        LOSS_LOG,
        "loss csv",
    )
    src = replace_once(
        src,
        '                        save_path = os.path.join(args.output_dir, '
        'f"checkpoint-{global_step}")\n'
        '                        accelerator.save_state(save_path)\n',
        SAVE_LORA,
        "lora checkpoint save",
    )
    src = replace_once(
        src,
        '        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(\n'
        '            args.pretrained_model_name_or_path,\n'
        '            text_encoder=unwrap_model(text_encoder),\n'
        '            vae=unwrap_model(vae),\n'
        '            unet=unwrap_model(unet),\n'
        '            revision=args.revision,\n'
        '            variant=args.variant,\n'
        '        )\n'
        '        pipeline.save_pretrained(args.output_dir)\n',
        FINAL_SAVE,
        "final save",
    )

    src = src.replace(
        "#!/usr/bin/env python\n", "#!/usr/bin/env python\n", 1
    )
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("import logging"):
            lines.insert(i, "import sys  # ADDED")
            lines.insert(i, "from pathlib import Path  # ADDED")
            break
    src = "\n".join(lines)

    head, _, rest = src.partition("\n")
    src = head + "\n" + HEADER + rest

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(src, encoding="utf-8")

    import ast
    ast.parse(src)

    added = sum(1 for ln in src.split("\n") if "# ADDED" in ln)
    modified = sum(1 for ln in src.split("\n") if "# MODIFIED" in ln)
    print(f"wrote {TARGET}")
    print(f"  {added} lines marked # ADDED, {modified} marked # MODIFIED")
    print(f"  syntax check passed")
    print(f"\nReview the diff with:\n  diff -u {BASELINE} {TARGET}")

if __name__ == "__main__":
    sys.exit(main())
