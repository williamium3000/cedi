import tqdm
import os
import json
import zipfile
import urllib.request

# Raw JSON files from the official Visual Genome website
VG_BASE_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
VG_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".vg_cache")

_VG_FILES = {
    "objects":       "objects.json.zip",
    "attributes":    "attributes.json.zip",
    "relationships": "relationships.json.zip",
    "regions":       "region_descriptions.json.zip",
    "image_data":    "image_data.json.zip",
}

def _download_and_load(name: str) -> list:
    os.makedirs(VG_CACHE_DIR, exist_ok=True)
    json_name = _VG_FILES[name].replace(".zip", "")
    json_path = os.path.join(VG_CACHE_DIR, json_name)

    # If plain JSON already present, use it directly
    if not os.path.exists(json_path):
        zip_path = os.path.join(VG_CACHE_DIR, _VG_FILES[name])
        if not os.path.exists(zip_path):
            url = f"{VG_BASE_URL}/{_VG_FILES[name]}"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(VG_CACHE_DIR)
        # Clean up zip after extraction to save space
        os.remove(zip_path)

    print(f"Loading {json_name} ...")
    with open(json_path) as f:
        return json.load(f)

# Build a unified list aligned with _objects_raw order
class _ListView:
    """Thin wrapper so existing code can use objects[idx], len(objects) etc."""
    def __init__(self, data): self._data = data
    def __len__(self):        return len(self._data)
    def __getitem__(self, i): return self._data[i]

def _build_objects_compat():
    rows = []
    for rec in _objects_raw:
        iid = rec["image_id"]
        w, h = _size_by_id.get(iid, (1, 1))
        rows.append({
            "image_id":  iid,
            # Prefer image_data.json's url — objects.json's image_url is buggy
            # (always points to VG_100K_2/, but ~half of VG images live in VG_100K/).
            "image_url": _url_by_id.get(iid, "") or rec.get("image_url", ""),
            "width":     w,
            "height":    h,
            "objects":   rec["objects"],
        })
    return rows

def _build_attributes_compat():
    rows = []
    for rec in _objects_raw:
        iid  = rec["image_id"]
        rows.append({"image_id": iid,
                     "attributes": _attr_by_id.get(iid, {}).get("attributes", [])})
    return rows

def _build_relationships_compat():
    rows = []
    for rec in _objects_raw:
        iid = rec["image_id"]
        rows.append({"image_id": iid,
                     "relationships": _rel_by_id.get(iid, {}).get("relationships", [])})
    return rows

def _build_regions_compat():
    rows = []
    for rec in _objects_raw:
        iid = rec["image_id"]
        rows.append({"image_id": iid,
                     "regions": _reg_by_id.get(iid, {}).get("regions", [])})
    return rows

objects = attributes = relationships = regions = None


def _ensure_loaded():
    """Load Visual Genome (~2.2 GB download, ~13 GB in memory) on first use rather than at
    import, so runs on other datasets (e.g. --dataset svg) do not pay for it."""
    global _objects_raw, _attr_by_id, _rel_by_id, _reg_by_id, _size_by_id, _url_by_id
    global objects, attributes, relationships, regions
    if objects is not None:
        return
    print("Loading Visual Genome data ...")
    _objects_raw       = _download_and_load("objects")        # list of {image_id, image_url, objects:[...]}
    _attributes_raw    = _download_and_load("attributes")
    _relationships_raw = _download_and_load("relationships")
    _regions_raw       = _download_and_load("regions")
    _image_data_raw    = _download_and_load("image_data")     # list of {image_id, width, height, url}

    # Index by image_id for O(1) lookup, preserving original list order
    _attr_by_id  = {r["image_id"]: r for r in _attributes_raw}
    _rel_by_id   = {r["image_id"]: r for r in _relationships_raw}
    _reg_by_id   = {r.get("image_id", r.get("id")): r for r in _regions_raw}
    _size_by_id  = {r["image_id"]: (r["width"], r["height"]) for r in _image_data_raw}
    _url_by_id   = {r["image_id"]: r.get("url", "") for r in _image_data_raw}

    objects       = _ListView(_build_objects_compat())
    attributes    = _ListView(_build_attributes_compat())
    relationships = _ListView(_build_relationships_compat())
    regions       = _ListView(_build_regions_compat())

    assert len(objects) == len(attributes) == len(relationships) == len(regions)

def load_sample_vg(idx):
    _ensure_loaded()
    import copy
    raw_sample = objects[idx]
    obj_list = raw_sample["objects"]
    attrs_list = attributes[idx]["attributes"]
    rels_raw = relationships[idx]["relationships"]
    regs = regions[idx]["regions"]

    # Raw VG: attributes.json is not 1:1 with objects.json. Use objects.json as
    # the authoritative object set; merge attributes by object_id when present.
    attrs_by_id = {a["object_id"]: a for a in attrs_list if "object_id" in a}
    cur_objects = {}
    for i, obj in enumerate(obj_list):
        original_id = obj["object_id"]
        merged = dict(obj)
        attr_entry = attrs_by_id.get(original_id)
        merged["attributes"] = attr_entry.get("attributes", []) if attr_entry else []
        merged["object_id"] = i
        cur_objects[original_id] = merged

    new_rels = []
    for rel_orig in rels_raw:
        sub = rel_orig.get("subject", {})
        obj = rel_orig.get("object", {})
        if sub.get("object_id") not in cur_objects or obj.get("object_id") not in cur_objects:
            continue
        rel = dict(rel_orig)
        rel.pop("relationship_id", None)
        rel["subject"] = cur_objects[sub["object_id"]]
        rel["object"] = cur_objects[obj["object_id"]]
        new_rels.append(rel)

    scene_graph = {
        "objects": cur_objects,
        "relationships": new_rels,
        "regions": regs,
    }
    # Build a fresh sample dict (do NOT mutate the cached raw_sample).
    sample = {k: v for k, v in raw_sample.items() if k != "objects"}
    sample["sg"] = scene_graph
    sample["metadata"] = {
        "objects":       copy.deepcopy(obj_list),
        "relationships": copy.deepcopy(rels_raw),
        "regions":       copy.deepcopy(regs),
        "attributes":    copy.deepcopy(attrs_list),
    }
    sample["image"] = _get_vg_image(sample["image_id"], sample.get("image_url", ""))
    return sample


_VG_IMAGE_CACHE_DIR = os.path.join(VG_CACHE_DIR, "images")


def _get_vg_image(image_id, url):
    """Return a PIL Image for the VG image, downloading + caching on first access.
    Falls back to the alternate VG_100K/VG_100K_2 path if the listed URL 404s.
    """
    from PIL import Image
    os.makedirs(_VG_IMAGE_CACHE_DIR, exist_ok=True)
    path = os.path.join(_VG_IMAGE_CACHE_DIR, f"{image_id}.jpg")
    if not os.path.exists(path):
        candidates = [url] if url else []
        if url:
            if "/VG_100K_2/" in url:
                candidates.append(url.replace("/VG_100K_2/", "/VG_100K/"))
            elif "/VG_100K/" in url:
                candidates.append(url.replace("/VG_100K/", "/VG_100K_2/"))
        last_err = None
        for u in candidates:
            try:
                urllib.request.urlretrieve(u, path)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"VG image_id={image_id} download failed: {last_err}")
    return Image.open(path).convert("RGB")


def load_vg(num_samples=None):
    _ensure_loaded()
    all_img_ids = range(len(objects))
    if num_samples is not None:
        all_img_ids = all_img_ids[:num_samples]

    samples = []
    print(f"loading vg: total {len(all_img_ids)}")
    for img_id in tqdm.tqdm(all_img_ids):
        case = load_sample_vg(img_id)
        samples.append(case)
    return samples

def format_case_vg(case, use_region=False):
    formatted = "Instances:\n"
    H = case["height"]
    W = case["width"]
    
    sg = case["sg"]

    for ori_id, ins in sg["objects"].items():
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        if ins.get("attributes", []) is None or len(ins.get("attributes", [])) == 0:
            cur_attr = "none"
        else:
            attrs = ins.get("attributes", [])
            cur_attr = ", ".join(attrs)
        formatted += f"instance {object_id}, {ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    
    formatted += "\nRelation between the above instances:\n"
    for rel in sg["relationships"]:
        formatted += f"{rel['subject']['names'][0]} (instance {rel['subject']['object_id']}) {rel['predicate'].lower()} {rel['object']['names'][0]} (instance {rel['object']['object_id']})\n"
    
    if use_region:
        formatted += "\nRegion descriptions:\n"
        for reg in case["metadata"]["regions"]:
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            formatted += f"description: {reg['phrase']}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})\n"
    
    return formatted
# Noisy VG object names that are attribute fragments, not real objects
NOISY_OBJECT_NAMES = frozenset({
    "color", "colour", "colors", "colours",
})


def format_case_vg_compact(case, use_region=False):
    """Compact VG formatter for v19 — same info, ~50% fewer tokens.

    Changes vs format_case_vg:
    - Drop 'instance' prefix: '0: clock [.53,.15,.62,.72] green, tall'
    - 2-decimal bbox in bracket notation, no 'bbox:' label
    - Omit attributes field entirely when empty (instead of 'attributes: none')
    - Filter out noisy pseudo-objects (color/colour)
    - Deduplicate relations
    - Relations use IDs only: '0 on 1' (names already in Objects section)
    """
    H = case["height"]
    W = case["width"]
    sg = case["sg"]

    formatted = "Objects:\n"
    valid_ids = set()
    for ori_id, ins in sg["objects"].items():
        object_id = ins["object_id"]
        name = ins["names"][0]
        if name.lower() in NOISY_OBJECT_NAMES:
            continue
        valid_ids.add(object_id)
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        attrs = ins.get("attributes", [])
        if attrs:
            formatted += f"{object_id}: {name} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}] {', '.join(attrs)}\n"
        else:
            formatted += f"{object_id}: {name} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]\n"

    formatted += "\nRelations:\n"
    seen_rels = set()
    for rel in sg["relationships"]:
        sub_id = rel['subject']['object_id']
        obj_id = rel['object']['object_id']
        pred = rel['predicate'].lower()
        if sub_id not in valid_ids or obj_id not in valid_ids:
            continue
        key = (sub_id, pred, obj_id)
        if key in seen_rels:
            continue
        seen_rels.add(key)
        formatted += f"{sub_id} {pred} {obj_id}\n"

    if use_region:
        formatted += "\nRegions:\n"
        for reg in case["metadata"]["regions"]:
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            formatted += f"{reg['phrase']} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]\n"

    return formatted


if __name__ == "__main__":
    case = load_vg(num_samples=5)[0]
    print("=== ORIGINAL ===")
    orig = format_case_vg(case)
    print(orig)
    print(f"Chars: {len(orig)}, ~Tokens: {len(orig)//4}")
    print()
    print("=== COMPACT ===")
    comp = format_case_vg_compact(case)
    print(comp)
    print(f"Chars: {len(comp)}, ~Tokens: {len(comp)//4}")
    print(f"Reduction: {(1 - len(comp)/len(orig))*100:.1f}%")
