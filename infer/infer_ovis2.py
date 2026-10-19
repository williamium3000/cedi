from ast import arg
import torch
import os
import argparse
import json
import tqdm

from transformers import AutoModelForCausalLM
import torch

def eval_model(model, image_file, query):
    query = f'<image>\n{query}'
    max_partition = 9
    images = [image_file]
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    
    
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="AIDC-AI/Ovis2-8B")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
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
        output = eval_model(model, image_file, q)
        
        output = output.strip().replace(".", '').lower()
        sample["output"] = output
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
            



