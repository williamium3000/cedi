# transformers version 4.43.0

import argparse
import json
import os
import torch
import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def eval_model(model, processor, image_file, query):
    # image = Image.open(image_file)
    image = image_file
    data = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompts = processor.apply_chat_template(data, add_generation_prompt=True)
    inputs = processor(prompts, image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="HuggingFaceM4/idefics2-8b")
    parser.add_argument('--checkpoints', type=str, default="checkpoints/idefics2-8b-lpoi-list5-10k/final/")
    args = parser.parse_args()

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        load_in_8bit=False,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)
    model.load_adapter(args.checkpoints)

    samples = json.load(open(args.infile, "r"))
    for sample in tqdm.tqdm(samples):
        query = sample['question']
        image_file = os.path.join(args.img_dir, sample["image"])
        output = eval_model(model, processor, image_file, query)
        sample["output"] = output

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)