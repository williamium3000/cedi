import torch
import os
import argparse
import json
import tqdm
from PIL import Image
import copy
import requests
from io import BytesIO


import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_model(processor, model, image_file, query, conversation_history=None):
    # Treat both None and empty history as "first turn".
    is_first_turn = not conversation_history
    return_tuple = conversation_history is not None  # wrapper unpacks a tuple

    if is_first_turn:
        # First round: image goes in the only user message; image features get
        # bound to this <image> token by the processor below.
        conversation_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
    else:
        # Later rounds: append a text-only user turn. The first user turn in
        # the history still carries {"type": "image"}, so apply_chat_template
        # produces a prompt with one <image> token; the processor below will
        # re-bind features from image_file to that position every turn.
        conversation_history = copy.deepcopy(conversation_history)
        conversation_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            }
        )
    
    prompt = processor.apply_chat_template(conversation_history, add_generation_prompt=True)
    # The first user message always carries {"type": "image"}, so the prompt
    # will contain one <image> token; the processor binds image features from
    # image_file to that position. We re-feed image_file every turn — LLaVA
    # cannot carry vision features in tokenized history alone.
    inputs = processor(images=image_file, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    output_text = processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

    if return_tuple:
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}],
        })
        return output_text, conversation_history
    return output_text





