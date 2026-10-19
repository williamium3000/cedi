import torch
import os
import argparse
import json
import tqdm
from PIL import Image

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    # raw_image = Image.open(image_file)
    inputs = processor(images=image_file, text=query, padding='longest', do_convert_rgb=True, return_tensors="pt").to(device)
    inputs = inputs.to(dtype=model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    output = processor.decode(output[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="google/paligemma-3b-mix-224")
    args = parser.parse_args()

    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    processor = PaliGemmaProcessor.from_pretrained(args.model_path)
    model.to(device)

    samples = json.load(open(args.infile, "r"))

    for sample in tqdm.tqdm(samples):
        q = sample["question"]
        image_file = os.path.join(args.img_dir, sample["image"])
        output = eval_model(processor, model, image_file, q)

        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
