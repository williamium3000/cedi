import torch
import os
import argparse
import json
import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def eval_model(processor, model, image_file, query):
    # Match reference: min_pixels/max_pixels for image; use process_vision_info + processor()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                },
                {"type": "text", "text": query},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # process_vision_info: (images, videos, video_kwargs) or (images, videos)
    try:
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except (ValueError, TypeError):
        images, videos = process_vision_info(messages, image_patch_size=16)
        video_kwargs = {}

    video_metadatas = None
    if videos is not None and len(videos) > 0:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        videos = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL inference (image + question -> output)")
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    # AutoModelForImageTextToText loads the correct class (qwen3_vl / qwen3_vl_moe) by config
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    samples = json.load(open(args.infile, "r"))

    for sample in tqdm.tqdm(samples):
        q = sample["question"]
        image_file = os.path.join(args.img_dir, sample["image"]) if args.img_dir else sample["image"]
        output = eval_model(processor, model, image_file, q)
        output = output.strip().replace(".", "").lower()
        sample["output"] = output

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
