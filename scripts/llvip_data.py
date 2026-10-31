"""Load cached LLVIP image pairs for InstructPix2Pix training.

Visible images are conditioning inputs and infrared images are targets. Shared
preprocessing keeps the inputs aligned across both runs.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROMPT = "turn this visible-light photo into an infrared thermal image"


def _to_tensor(img):
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    arr = 2.0 * (arr / 255.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class LLVIPPairs(Dataset):
    def __init__(self, data_root, manifest, tokenizer, prompt=PROMPT, random_flip=False):
        self.root = Path(data_root)
        self.cache = self.root / "cache"
        self.entries = [
            line.strip() for line in (self.root / manifest).read_text().splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.random_flip = random_flip

        if not self.entries:
            raise ValueError(f"{manifest} is empty")

        missing = []
        for rel in self.entries[:20]:
            split, name = rel.split("/", 1)
            if not (self.cache / split / "visible" / name).exists():
                missing.append(rel)
        if missing:
            raise FileNotFoundError(
                f"cache missing for {missing[:3]} - run prepare_llvip.py first"
            )

        self.input_ids = tokenizer(
            [prompt],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

    def __len__(self):
        return len(self.entries)

    def names(self):
        return [e.split("/", 1)[1] for e in self.entries]

    def __getitem__(self, idx):
        rel = self.entries[idx]
        split, name = rel.split("/", 1)

        visible = Image.open(self.cache / split / "visible" / name)
        infrared = Image.open(self.cache / split / "infrared" / name)

        if self.random_flip and torch.rand(1).item() < 0.5:
            visible = visible.transpose(Image.FLIP_LEFT_RIGHT)
            infrared = infrared.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            "original_pixel_values": _to_tensor(visible),
            "edited_pixel_values": _to_tensor(infrared),
            "input_ids": self.input_ids,
            "name": name,
        }


def collate_fn(examples):
    return {
        "original_pixel_values": torch.stack(
            [e["original_pixel_values"] for e in examples]
        ).to(memory_format=torch.contiguous_format).float(),
        "edited_pixel_values": torch.stack(
            [e["edited_pixel_values"] for e in examples]
        ).to(memory_format=torch.contiguous_format).float(),
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
        "names": [e["name"] for e in examples],
    }


def load_fixed_batch(data_root, resolution=256):
    root = Path(data_root)
    rels = [line.strip() for line in (root / "val_fixed.txt").read_text().splitlines()
            if line.strip()]
    images, names = [], []
    for rel in rels:
        split, name = rel.split("/", 1)
        images.append(Image.open(root / "cache" / split / "visible" / name).convert("RGB"))
        names.append(name)
    return images, names


def ground_truth_infrared(data_root, split, name):
    return Image.open(Path(data_root) / "cache" / split / "infrared" / name).convert("RGB")
