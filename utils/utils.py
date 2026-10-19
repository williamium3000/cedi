from .vg import load_vg
from .svg import load_svg

import os
from PIL import Image
import argparse
import torch
import json
import requests
from io import BytesIO
import re
import tqdm


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_data(args):
    if args.dataset == "vg":
        samples = load_vg(args.num_samples)
    elif args.dataset == "svg":
        print("Loading SVG dataset (HuggingFace, VG-format with images)...")
        samples = load_svg(num_samples=args.num_samples)
        print(f"Loaded {len(samples)} SVG samples")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}. Supported: vg, svg.")
    return samples
