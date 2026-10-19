import torch
import os
import argparse
import json
import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import copy

def eval_model(processor, model, image_file, query, conversation_history=None):
    # Track whether conversation_history was originally None to determine return type
    was_none = conversation_history is None
    
    if conversation_history is not None and len(conversation_history) > 0:
        conversation_history = copy.deepcopy(conversation_history)
        conversation_history.append(
            {
            "role": "user",
            "content": [
                {"type": "text", "text": query}
            ]
        }
        )

    else:
        # Initialize conversation_history as a new list if it's None or empty
        conversation_history = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_file},
                            {"type": "text", "text": query}
                        ]
                    }
                ]
    messages = conversation_history
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generation = generation[0][input_len:]

    output_text = processor.decode(generation, skip_special_tokens=True)
    
    # Return tuple only if conversation_history was originally provided (not None)
    if not was_none:
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}]
        })
        return output_text, conversation_history
    else:
        return output_text




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="google/gemma-3-27b-it")
    args = parser.parse_args()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    
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
            



