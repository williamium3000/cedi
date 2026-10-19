import torch
import os
import argparse
import json
import tqdm
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    
    # raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image_file, text=query, return_tensors="pt").to(device)
    
    output = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    output = processor.batch_decode(output, skip_special_tokens=True)[0]
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="Salesforce/instructblip-vicuna-7b",
                        choices=[
                            "Salesforce/instructblip-vicuna-7b"
                            ])
    args = parser.parse_args()
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path)
    processor = InstructBlipProcessor.from_pretrained(args.model_path)
    model.to(device)
    
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
        image_file = os.path.join(args.img_dir, sample["image"])
        output = eval_model(processor, model, image_file, q)
        
        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
            



