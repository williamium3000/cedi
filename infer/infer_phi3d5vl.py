import torch
import os
import argparse
import json
import tqdm
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    
    # raw_image = Image.open(image_file).convert("RGB")
    images = [image_file]
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{query}"},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 1024, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(
        **inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    print(output)
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="microsoft/Phi-3.5-vision-instruct",
                        choices=[
                            "microsoft/Phi-3.5-vision-instruct"
                            ])
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map=device, 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2'    
        )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(args.model_path, 
        trust_remote_code=True, 
        num_crops=4
        ) 

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
            



