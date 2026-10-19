import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

from torchvision import transforms

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from PIL import Image
import json

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "infer/eval_configs/minigpt4_eval.yaml",
    "instructblip": "infer/eval_configs/instructblip_eval.yaml",
    "shikra": "infer/eval_configs/shikra_eval.yaml",
    "llava-1.5": "infer/eval_configs/llava-1.5_eval.yaml",
}


INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def eval_model(image_file, model, image_processor, query, args):
    # image = Image.open(image_file).convert("RGB")
    image = image_file
    
    image = image_processor["eval"](image).unsqueeze(0)
    image = image.to(device)

    template = INSTRUCTION_TEMPLATE[args.model]
    query = template.replace("<question>", query)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)


    with torch.inference_mode():
        with torch.no_grad():
            outputs = model.generate(
                {"image": norm(image), "prompt": query},
                use_nucleus_sampling=False,
                num_beams=5,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=50,
                threshold=15.0,
                num_attn_candidates=5,
            )
    return outputs[0]


if __name__ == "__main__":
    pass