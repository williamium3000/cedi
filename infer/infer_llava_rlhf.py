from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    DEFAULT_IMAGE_PATCH_TOKEN
)
from llava.model import *
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token
)

import argparse
import torch
import json
import re
import os
import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO



def eval_model(tokenizer, model, image_processor, image_file, query):
    disable_torch_init()

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = [image_file]
    # images = [args.image_file.convert("RGB")]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    # ✅ force [1, T]
    if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device)

    assert input_ids is not None
    assert input_ids.dim() == 2, f"bad input_ids shape: {tuple(input_ids.shape)}"

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def load_pretrained_model(model_path, lora_path, model_name, load_bf16=False):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        model,
        lora_path,
    )
    model = model.merge_and_unload()
    print('Convert to FP16...')
    model.to(torch.float16)


    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    if load_bf16:
        vision_tower.to(device='cuda', dtype=torch.bfloat16)
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="llava-rlhf-13b-v1.5-336")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="LLaVA-RLHF-13b-v1.5-336/sft_model")
    parser.add_argument("--lora-path", type=str, default="LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model")
    parser.add_argument("--load_bf16", action="store_true")
    
    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.lora_path, args.model_name, args.load_bf16)

    model_path = args.model_path
    model_name = args.model_name

    samples = json.load(open(args.infile, "r"))

    for sample in tqdm.tqdm(samples):
        q = sample["question"]
        
        # Check if image is a URL
        if sample["image"].startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(sample["image"], timeout=10)
            response.raise_for_status()
            image_file = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load image from local path
            image_file = os.path.join(args.img_dir, sample["image"])
            image_file = Image.open(image_file).convert('RGB')
        
        output = eval_model(tokenizer, model, image_processor, context_len, type('Args', (), {
                                "model_path": model_path,
                                "model_base": None,
                                "model_name": model_name,
                                "query": q,
                                "conv_mode": None,
                                "image_file": image_file,
                                "sep": ",",
                                "load_in_8bit": False,
                                "load_in_4bit": False,
                                "temperature": 0.0,  # set as 0.0 for reproceduce
                                "top_p": None,
                                "num_beams": 1,
                                "max_new_tokens": 2048
                            })())
        output = output.strip().replace(".", '').lower()
        print(output)
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    json.dump(samples, open(args.outfile, "w"), indent=4)
            



