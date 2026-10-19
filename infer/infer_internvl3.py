import math
import torch
import os
import argparse
import json
import tqdm
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import requests
from io import BytesIO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def eval_model(model, tokenizer, image_file, query, conversation_history=None):
    """
    Evaluate model with support for multi-turn conversations.
    Referenced from: https://huggingface.co/OpenGVLab/InternVL3-2B-Pretrained
    
    Args:
        model: The InternVL3 model
        tokenizer: The tokenizer
        image_file: Path to the image file or PIL Image
        query: The user query/question
        conversation_history: Optional conversation history from previous turns.
                            If None, starts a new conversation.
    
    Returns:
        str or tuple: If conversation_history was None, returns just the response string.
                     Otherwise, returns (response, updated_conversation_history).
    """
    # Treat both None and empty history as "first turn" — the wrapper passes []
    # on first call and the only previously-handled case was `is None`, which
    # silently dropped the image on round 0.
    is_first_turn = not conversation_history
    return_tuple = conversation_history is not None  # wrapper unpacks a tuple

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # Encode the image every turn. The official docs suggest passing
    # pixel_values=None on later turns, but InternVLChatModel.generate() only
    # patches vision features into IMG_CONTEXT_TOKEN positions when
    # pixel_values is non-None — otherwise the model does text-only generation
    # despite the history containing an <image> token from turn 0. So we
    # always re-feed pixel_values to keep the image visible across turns.
    pixel_values = load_image(image_file, max_num=12).to(torch.bfloat16).cuda()

    if is_first_turn:
        # Round 0: prepend <image> to the question; chat() will substitute it
        # with the image-token block before generation.
        query_with_image = f'<image>\n{query}'
        response, conversation_history = model.chat(
            tokenizer, pixel_values, query_with_image, generation_config,
            history=None, return_history=True,
        )
    else:
        # Later rounds: history already contains <image> in turn 0; chat() only
        # replaces the first <image> occurrence, so the new question must NOT
        # contain another <image> token.
        response, conversation_history = model.chat(
            tokenizer, pixel_values, query, generation_config,
            history=conversation_history, return_history=True,
        )

    if return_tuple:
        return response, conversation_history
    return response

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="OpenGVLab/InternVL3-1B")
    args = parser.parse_args()
    
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    device_map = split_model(args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    
    # format:
    # a list of dict
    # minimum keys: image, question
    # [
    #     {
    #         "image": "xxxx",
    #         "question": "xxxxx"},
    #     ...]
    # leave the output key empty
    
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
        
        output = eval_model(model, tokenizer, image_file, q)
        
        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
