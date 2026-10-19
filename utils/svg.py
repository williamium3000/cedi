"""
Synthetic Visual Genome (SVG) Dataset Loader
Loads the SVG dataset in VG format from HuggingFace: nnonymous/svg5000_in_vg

The dataset has been pre-converted to VG format with 'sg' key for direct use with SceneGraphData.
"""

from datasets import load_dataset
import tqdm
import json
from typing import List, Dict, Optional


def load_svg(num_samples: Optional[int] = None, split: str = "train_500_augmented") -> List[Dict]:
    """
    Load SVG dataset in VG format.

    Loads from nnonymous/svg5000_in_vg which has pre-converted SVG data to VG format
    with 'sg' key compatible with SceneGraphData.

    Args:
        num_samples: Number of samples to load (None = all)
        split: HF split to load. "train" = original noun-phrase names with empty
            attributes; "train_500_augmented" (default) = first 500 rows with
            cleaned head-noun names and OpenAI-extracted attributes.

    Returns:
        List of sample dictionaries with 'sg' key in VG format, images, and metadata.
    """
    print(f"Loading SVG dataset in VG format from nnonymous/svg5000_in_vg [{split}]...")
    dataset = load_dataset("nnonymous/svg5000_in_vg", split=split)
    
    # Only load the requested number of samples
    num_to_load = num_samples if num_samples is not None else len(dataset)
    num_to_load = min(num_to_load, len(dataset))
    
    all_samples = []
    print(f"Processing {num_to_load} samples...")
    for idx in tqdm.tqdm(range(num_to_load)):
        sample = dict(dataset[idx])

        # Parse sg if it's a JSON string (deserialize from HuggingFace format)
        if 'sg' in sample and isinstance(sample['sg'], str):
            try:
                sample['sg'] = json.loads(sample['sg'])
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse sg for sample {idx}")

        # Parse scene_graph if it's a JSON string (for reference)
        if 'scene_graph' in sample and isinstance(sample['scene_graph'], str):
            try:
                sample['scene_graph'] = json.loads(sample['scene_graph'])
            except json.JSONDecodeError:
                pass

        _normalize_svg_sample(sample)
        all_samples.append(sample)

    print(f"Loaded {len(all_samples)} SVG samples in VG format")
    return all_samples


def _normalize_svg_sample(sample: dict) -> None:
    """Mutate an SVG sample in-place to match the VG schema expected by all consumers.

    Raises ValueError if any required field is absent or empty so that bad data
    is caught at load time rather than causing silent downstream errors.
    """
    from nltk.corpus import wordnet as _wn

    image_id = sample.get('image_id')
    if not image_id:
        raise ValueError(f"sample missing 'image_id': {list(sample.keys())}")

    sg = sample.get('sg')
    if not isinstance(sg, dict):
        raise ValueError(f"sample {image_id!r}: 'sg' must be a dict, got {type(sg)}")

    # --- url: provided by HF dataset; raise if absent or empty ---------------
    url = sample.get('url')
    if not url:
        raise ValueError(f"sample {image_id!r}: missing 'url' — ensure the HF dataset "
                         "has the url column populated for all rows")

    # --- coco_id: numeric stem for COCO/VG images; None for ADE20K -----------
    stem = image_id.split('/')[-1].split('.')[0]
    try:
        sample['coco_id'] = int(stem)
    except ValueError:
        sample['coco_id'] = None  # ADE20K and similar non-numeric ids

    # --- Object name normalisation + synset population ------------------------
    objects_list = list(sg.get('objects', {}).values())
    if not objects_list:
        raise ValueError(f"sample {image_id!r}: sg['objects'] is empty")

    for obj in objects_list:
        raw_name = (obj.get('names') or [''])[0]
        if not raw_name:
            raise ValueError(f"sample {image_id!r}: object {obj.get('object_id')} has no name")
        head = raw_name.split()[-1]
        obj['names'] = [head]
        if not obj.get('synsets'):
            obj['synsets'] = [s.name() for s in _wn.synsets(head, pos=_wn.NOUN)[:3]]

    # Relationship subject/object dicts are independent JSON-decoded copies.
    for rel in sg.get('relationships', []):
        for side in ('subject', 'object'):
            entity = rel.get(side)
            if entity:
                raw = (entity.get('names') or [''])[0]
                if raw:
                    entity['names'] = [raw.split()[-1]]

    # --- sg['regions']: VG-style phrase regions for CHAIR / HaELM -----------
    hf_regions = sample.get('regions')
    if hf_regions is None:
        raise ValueError(f"sample {image_id!r}: missing 'regions' column from HF dataset")
    phrase_regions = [
        {"phrase": r['object'].split(' in ')[0].strip()}
        for r in hf_regions
        if r.get('object')
    ]
    sg['regions'] = phrase_regions

    # --- metadata dict: mirrors VG's sample['metadata'] layout ---------------
    attributes_list = [
        {
            'object_id': obj.get('object_id'),
            'names': obj.get('names', []),
            'synsets': obj.get('synsets', []),
            'attributes': obj.get('attributes') or [],
        }
        for obj in objects_list
    ]
    sample['metadata'] = {
        'objects': objects_list,
        'attributes': attributes_list,
        'relationships': sg.get('relationships', []),
        'regions': phrase_regions,
    }


def format_case_svg(case: Dict, use_region: bool = False) -> str:
    """
    Format SVG sample in VG style for display/evaluation.
    
    Args:
        case: Sample dictionary from SVG (with 'sg' key in VG format)
        use_region: Whether to include region descriptions (not used for SVG)
    
    Returns:
        Formatted string representation matching VG format
    """
    formatted = "Instances:\n"
    H = case.get("height", 1000)
    W = case.get("width", 1000)
    
    sg = case.get("sg", {})
    
    # Format objects
    for ori_id, ins in sg.get("objects", {}).items():
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        
        if ins.get("attributes", []) is None or len(ins.get("attributes", [])) == 0:
            cur_attr = "none"
        else:
            attrs = ins.get("attributes", [])
            cur_attr = ", ".join(attrs)
        
        formatted += f"instance {object_id}, {ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    
    # Format relationships
    formatted += "\nRelation between the above instances:\n"
    for rel in sg.get("relationships", []):
        formatted += f"{rel['subject']['names'][0]} (instance {rel['subject']['object_id']}) {rel['predicate'].lower()} {rel['object']['names'][0]} (instance {rel['object']['object_id']})\n"
    
    return formatted


if __name__ == "__main__":
    print(format_case_svg(load_svg(num_samples=5)[0]))

# creating the SVG dataset in VG format involved several key steps to ensure compatibility with our existing pipelines and to enhance the quality of the data for evaluation:
# 1. Object names — 1 word everywhere (already being fixed)

# sg.objects[*].names → exactly 1 word (last word of compound noun)
# sg.relationships[*].subject.names and sg.relationships[*].object.names → same, must also be 1 word (these are separate JSON dicts, not references to sg.objects)
# 2. Sampling (already being changed)

# 100 VG + 200 COCO + 200 ADE
# 3. Predicate normalization (new)

# Current SVG predicates are verbose: "used for sitting at dining table", "provides access to", "located near", "placed on"
# VG uses short tokens: "on", "near", "at", "wears", "next to" (≤3 words)
# Normalize sg.relationships[*].predicate to ≤3 words
# 4. Add a url column (new — for FaithScore grader)

# Add a url field to every row with a direct HTTP download link to the image
# COCO ("train2017/000000554750.jpg"): already handled by our code → http://images.cocodataset.org/train2017/000000554750.jpg — but agent can include it explicitly
# VG Flickr ("2361014.jpg"): → https://cs.stanford.edu/people/rak248/VG_100K/2361014.jpg
# ADE20K: host images in the same HF repo (e.g. ade20k_images/ subfolder) and provide: https://huggingface.co/datasets/nnonymous/svg5000_in_vg/resolve/main/ade20k_images/ADE_train_00018793.jpg
# This URL is only read by grader/faithscore/eval.py — the examiner pipeline uses the HF image column directly
# 5. Region descriptions (optional, nice to have)

# Current regions[*].object is a single-word/short label ("bottle", "chair")
# HaELM uses these as reference captions — richer descriptions help (e.g. "a bottle on the counter")
# Requires LLM generation; skip if too costly