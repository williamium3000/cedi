import argparse
import json
import os
import re
from collections import defaultdict
import numpy as np
import networkx as nx
import nltk
# nltk.download('wordnet')
import tqdm
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from utils.llm import LLMChat
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import netrd
import netrd.distance

_wordnet_lock = threading.Lock()
lemmatizer = WordNetLemmatizer()


def merge_conversations_by_image_id(data):
    grouped = defaultdict(lambda: {
        'image_id': None,
        'url': None,
        'width': None,
        'height': None,
        'coco_id': None,
        'flickr_id': None,
        'sg': None,
        'conversations': [],
        'context': {"relevant_objects": []}
    })

    for item in data:
        image_id = item['image_id']

        if grouped[image_id]['image_id'] is None:
            grouped[image_id]['image_id'] = item['image_id']
            grouped[image_id]['url'] = item.get('url')
            grouped[image_id]['width'] = item.get('width')
            grouped[image_id]['height'] = item.get('height')
            grouped[image_id]['coco_id'] = item.get('coco_id')
            grouped[image_id]['flickr_id'] = item.get('flickr_id')
            grouped[image_id]['sg'] = item.get('sg')

        if 'conversations' in item and item['conversations']:
            grouped[image_id]['conversations'].extend(item['conversations'])

        if 'context' in item and item['context']:
            grouped[image_id]['context']["relevant_objects"].extend(
                item.get('context', {}).get("relevant_objects", [])
            )

    merged = list(grouped.values())
    merged.sort(key=lambda x: x['image_id'])
    return merged

def word_to_synset(word, pos_tag=wordnet.NOUN):
    word = ' '.join(word.strip().lower().split())
    lemma_synset = set()

    # If the word consists of multiple parts, join them with an underscore
    word_split = word.split()
    if len(word_split) >= 2:
        word = "_".join(word_split)

    # # Add all synsets of the word to the set
    with _wordnet_lock:
        synonyms = wordnet.synsets(word, pos=pos_tag)
        
    for sys in synonyms:
        for lemma in sys.lemmas():
            lemma_synset.add(lemma.synset())

    return set().union(*[lemma_synset])


def similar_to_any(candidate, reference):
    candidate_synsets = word_to_synset(candidate)
    ref_synsets = word_to_synset(reference)
    return 0 if candidate_synsets & ref_synsets else 1


def node_subst_cost(n1_attrs, n2_attrs):
    name1 = n1_attrs.get("name")
    name2 = n2_attrs.get("name")

    if "attributes" in n1_attrs:
        attributes1 = n1_attrs["attributes"]
    else:
        attributes1 = []

    if "attributes" in n2_attrs:
        attributes2 = n2_attrs["attributes"]
    else:
        attributes2 = []
    cost = len(set(attributes1) ^ set(attributes2)) + similar_to_any(name1, name2)
    return cost


def edge_subst_cost(e1_attrs, e2_attrs):
    predicates1 = set(e1_attrs.get("predicates", []))
    predicates2 = set(e2_attrs.get("predicates", []))
    cost = 0 if predicates1 & predicates2 else 1
    return cost


def compare_scene_graphs(gt_graph, pred_graph, timeout, method="ged"):
    if method == "ged":
        return nx.graph_edit_distance(
            gt_graph,
            pred_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            node_del_cost=lambda attrs: 1 + len(attrs.get("attributes", [])),
            node_ins_cost=lambda attrs: 1 + len(attrs.get("attributes", [])),
            edge_del_cost=lambda attrs: 1,
            edge_ins_cost=lambda attrs: 1,
            timeout=timeout
        )

    nodes1 = set(gt_graph.nodes())
    nodes2 = set(pred_graph.nodes())
    all_nodes = sorted(list(nodes1 | nodes2))
    N = len(all_nodes)

    def align_graph(original_g, full_node_list):
        new_g = nx.Graph()
        new_g.add_nodes_from(full_node_list)
        new_g.add_edges_from(original_g.to_undirected().edges())
        return new_g

    g1_aligned = align_graph(gt_graph, all_nodes)
    g2_aligned = align_graph(pred_graph, all_nodes)

    if method == "netlsd":
        g1 = gt_graph.to_undirected()
        g2 = pred_graph.to_undirected()

        if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return 1.0 if g1.number_of_nodes() != g2.number_of_nodes() else 0.0

        dist_obj = netrd.distance.NetLSD()
        topo_dist = dist_obj.dist(g1, g2)
        # topo_dist = np.tanh(topo_dist)
    elif method == "delta_con":
        dist_obj = netrd.distance.DeltaCon()
        topo_dist = dist_obj.dist(g1_aligned, g2_aligned)
        # topo_dist = np.tanh(raw_topo_dist)
    elif method == "portrait":
        dist_obj = netrd.distance.PortraitDivergence()
        topo_dist = dist_obj.dist(g1_aligned, g2_aligned)
    elif method == "hamming":
        dist_obj = netrd.distance.Hamming()
        raw_hamming = dist_obj.dist(g1_aligned, g2_aligned) / (len(all_nodes) ** 2)

        max_edges = N * (N - 1) if N > 1 else 1
        topo_dist = raw_hamming / max_edges
    elif method == "him":
        dist_obj = netrd.distance.HammingIpsenMikhailov()
        topo_dist = dist_obj.dist(g1_aligned, g2_aligned, combination_factor=1.0)
    elif method == "frobenius":
        dist_obj = netrd.distance.Frobenius()
        raw_frob = dist_obj.dist(g1_aligned, g2_aligned)
        max_frob = np.sqrt(N * (N - 1)) if N > 1 else 1.0
        topo_dist = raw_frob / max_frob
    elif method == "graph_diffusion":
        dist_obj = netrd.distance.GraphDiffusion()
        topo_dist = dist_obj.dist(g1_aligned, g2_aligned)
        topo_dist = np.log1p(topo_dist) / (np.log1p(N))
        topo_dist = min(topo_dist, 1.0)
    elif method == "jaccard":
        dist_obj = netrd.distance.JaccardDistance()
        topo_dist = dist_obj.dist(g1_aligned, g2_aligned)
    elif method == "netsimile":
        dist_obj = netrd.distance.NetSimile()
        raw_dist = dist_obj.dist(g1_aligned, g2_aligned)
        denom = np.log2(N + 2) * 5
        topo_dist = min(raw_dist / denom, 1.0)
    elif method == "resistance":
        dist_obj = netrd.distance.ResistancePerturbation()
        raw_res = dist_obj.dist(g1_aligned, g2_aligned, p=2)
        topo_dist = np.tanh(raw_res / (N ** 2)) if N > 0 else 0.0
    else:
        raise ValueError(f"Invalid method: {method}")
        
    total_attr_diff = 0
    total_attr_union = 0
    for node in all_nodes:
        attrs1 = set(gt_graph.nodes[node].get("attributes", [])) if node in nodes1 else set()
        attrs2 = set(pred_graph.nodes[node].get("attributes", [])) if node in nodes2 else set()

        total_attr_diff += len(attrs1 ^ attrs2)
        total_attr_union += len(attrs1 | attrs2)

        if node not in nodes1 or node not in nodes2:
            total_attr_diff += 1
            total_attr_union += 1

    attr_dist_norm = total_attr_diff / total_attr_union if total_attr_union > 0 else 0
    # total_attr_diff, attr_dist_norm = 0, 0

    final_score = topo_dist + attr_dist_norm if method in ["portrait", "delta_con", "him", "frobenius", "jaccard", "graph_diffusion", "netsimile"] else topo_dist + total_attr_diff
    print(f"topo_dist: {topo_dist}, attr_dist_norm: {attr_dist_norm}, total_attr_diff: {total_attr_diff} ")
    return float(final_score)


def add_node_with_attributes(graph, name, attributes):
    if name not in graph.nodes:
        if attributes:
            # if not isinstance(attributes, list):
            #     print('attributes type', attributes)
            graph.add_node(name, name=name, attributes=attributes)
        else:
            graph.add_node(name, name=name)
    elif attributes and len(attributes):
        if "attributes" not in graph.nodes[name]:
            graph.nodes[name]["attributes"] = []

        for att in attributes:
            if att not in graph.nodes[name]["attributes"]:
                graph.nodes[name]["attributes"].append(att)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def lemma_word(ori_word):
    tokens = ori_word.lower().split(' ')
    pos_tags = nltk.pos_tag(tokens)

    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)).lower() for word, tag in pos_tags]
    lemma_word = ' '.join(lemmas)
    return lemma_word


def scene_graph_to_nx(relationships, attributes, objects):
    graph = nx.DiGraph()

    # Index attributes by object_id
    if isinstance(attributes, dict):
        attr_map = {attr["object_id"]: {k: v for k, v in attr.items() if k != "object_id"}
                    for key, attr in attributes.items()}
    else:
        attr_map = {attr["object_id"]: {k: v for k, v in attr.items() if k != "object_id"}
                    for attr in attributes}

    # Add nodes edges from relationships
    for rel in relationships:
        subj = rel["subject"].copy()
        obj = rel["object"].copy()

        sub_name = lemma_word(subj["names"][0])
        obj_name = lemma_word(obj["names"][0])
        # Merge attributes if available
        if subj["object_id"] in attr_map.keys():
            subj.update(attr_map[subj["object_id"]])
        if obj["object_id"] in attr_map:
            obj.update(attr_map[obj["object_id"]])

        # Add nodes
        add_node_with_attributes(graph, sub_name, subj.get("attributes", []))
        add_node_with_attributes(graph, obj_name, obj.get("attributes", []))

        # Add edge
        pred = lemma_word(rel['predicate'])
        if graph.has_edge(sub_name, obj_name):
            if pred not in graph[sub_name][obj_name]["predicates"]:
                graph[sub_name][obj_name]["predicates"].append(pred)
        else:
            graph.add_edge(sub_name, obj_name, predicates=[pred])

    # Add attribute-only nodes
    for obj_id, attrs in attr_map.items():
        att_name = lemma_word(attrs["names"][0])
        if att_name not in graph.nodes and attrs["attributes"]:
            graph.add_node(att_name, name=att_name, attributes=attrs["attributes"])
        elif att_name not in graph.nodes:
            graph.add_node(att_name, name=att_name)
        elif attrs["attributes"] and len(attrs["attributes"]):
            if "attributes" not in graph.nodes[att_name]:
                graph.nodes[att_name]["attributes"] = []

            for att in attrs["attributes"]:
                if att not in graph.nodes[att_name]["attributes"]:
                    graph.nodes[att_name]["attributes"].append(att)
        # else:
        #     print(attrs)

    # Add object-only nodes
    if isinstance(objects,dict):
        for key, obj in objects.items():
            obj_name = lemma_word(obj["names"][0])
            if obj_name not in graph.nodes:
                graph.add_node(obj_name, name=obj_name)
    else:
        for obj in objects:
            obj_name = lemma_word(obj["names"][0])
            if obj_name not in graph.nodes:
                graph.add_node(obj_name, name=obj_name)

    return graph


def parse_scene_graph_string(s):
    triplets = []
    for match in re.findall(r"\((.*?)\)", s):
        parts = [p.strip() for p in match.split(",")]
        if len(parts) == 3:
            triplets.append(tuple(parts))
        elif len(parts) == 1:
            triplets.append((parts[0],))
        else:
            # Malformed triplet from LLM parser (e.g., 2 or 4+ parts); skip
            # rather than fail the whole run, since a small fraction of
            # rounds is expected to produce noisy SG output.
            print(f"[warn] skipping malformed triplet: ({match})")
            continue
    return triplets


def build_graph_from_string(s):
    graph = nx.DiGraph()
    triplets = parse_scene_graph_string(s)

    for triple in triplets:
        if len(triple) == 3:
            subj, pred, obj = triple
            subj, pred, obj = lemma_word(subj), lemma_word(pred), lemma_word(obj)
            if subj not in graph.nodes:
                graph.add_node(subj, name=subj)

            if pred == "be":
                if "attributes" not in graph.nodes[subj]:
                    graph.nodes[subj]["attributes"] = []
                if obj not in graph.nodes[subj]["attributes"]:
                    graph.nodes[subj]["attributes"].append(obj)
            else:
                if obj not in graph.nodes:
                    graph.add_node(obj, name=obj)

                if graph.has_edge(subj, obj):
                    if pred not in graph[subj][obj]["predicates"]:
                        graph[subj][obj]["predicates"].append(pred.lower())
                else:
                    graph.add_edge(subj, obj, predicates=[pred.lower()])
        else:
            node = triple[0]
            node = lemma_word(node)
            if node not in graph.nodes:
                graph.add_node(node, name=node)

    return graph


def parse_single_question(agent, question):
    """Parse a single question into scene graph triplets."""
    LSG_PROMPT = f"""From the given sentence, the task is to extract scene graphs formed as <subject, predicate, object>, <object, is, attribute> or <object>. Note that the subject is the physical entity or noun that performs the action or is being described, and the object is the physical entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). The attribute is a physical quality or characteristic (typically an adjective) directly modifying an object or entity (e.g., <jacket, is, red>, <wall, is, wooden>).
Instructions:
- If an object has no attributes or relations, output it directly in the form <object>.
- Do **not** extract scene graphs involving:
  - Objects, subjects or relations that are negated (e.g., "There is no man...")
  - Non-physical entities (e.g., "atmosphere", "conversation") in subject or object
  - Entities or relations that are **speculative or inferred** from other clues rather than explicitly described as visible (e.g., "could indicate", "might suggest", "possibly", "likely")
  - Abstract scene descriptions that cannot be directly grounded in physical objects or traits (e.g., "scene is urban", "shirt adds pop of color", "building contributes to atmosphere")
  - Attributes that are subjective or stylistic rather than physical (e.g., "beautiful", "cozy", "futuristic" when not tied to tangible features)
  - Statements about effects, purposes, or benefits rather than direct physical description (e.g., "contributes to convenience", "supports community well-being")

### Examples
Sentence: "A slice of bread is covered with a sour cream and guacamole."
Triplets: <bread, covered with, sour cream>, <bread, covered with, guacamole>

Sentence: "A beautiful woman walking a dog on top of a beach."
Triplets: <woman, walking with, dog>, <woman, on, beach>, <dog, on, beach>

Sentence: "Four clocks sitting on a floor next to a woman's feet."
Triplets: <clock, sitting on, floor>, <clock, next to, feet>

Sentence: "One person sits in a chair looking at her phone while another rests on the couch."
Triplets: <person, sits in, chair>, <person, looking at, phone>, <person, rests on, couch>

Sentence: "A lady and a child near a park bench with kites and ducks flying in the sky and on the ground."
Triplets: <lady, near, park bench>, <child, near, park bench>, <kites, flying in, sky>, <ducks, on, ground>

Sentence: "Two men sit on a bench near the sidewalk and one of them talks on a cell phone."
Triplets: <men, sit on, bench>, <bench, near, sidewalk>, <man, talks on, phone>

Sentence: "There is no man wearing a red jacket in the image."  
Triplets: (none)

Sentence: "A man wearing a red jacket is in the image."  
Triplets: <man, wearing, jacket>, <jacket, is, red>

Sentence: "The carpet on the wooden floor is blue."  
Triplets: <carpet, on, floor>, <floor, is, wooden>, <carpet, is, blue>

Sentence: "There are several cars parked along the street, and a bicycle is also visible."  
Triplets: <cars, parked on, street>, <bicycle>

Sentence: "People on the street are engaged in various activities and interactions."  
Triplets: <people, on, street>

Sentence: "The traffic light indicates that the street is regulated for vehicle and pedestrian safety."  
Triplets: <traffic light>, <street>

### Now extract triplets from the following sentence:
Sentence: \"{question}\"\nTriplets:
            """

    messages = [
        {"role": "system", "content": "From the given sentence, your task is to extract meaningful triplets formed as <subject, predicate, object>."},
        {"role": "user", "content": LSG_PROMPT.strip()}
    ]

    try:
        from grader.sg.llm_parser import refine_output
        parsing_result = agent.chat(messages, None)
        refined_sg = refine_output(parsing_result)
        return refined_sg
    except Exception as e:
        print(f"Error parsing question '{question[:50]}...': {e}")
        return []


def batch_parse_scene_graph(samples, agent, max_workers=10):
    """
    Batch parse scene graphs for multiple samples in parallel.
    
    Args:
        samples: List of samples to parse
        agent: LLMChat instance
        max_workers: Maximum number of parallel threads
    
    Returns:
        Dictionary mapping sample_idx -> (sg_dict, unique_sg)
    """
    # Collect all parsing tasks: (sample_idx, conv_idx, question_idx, question)
    parsing_tasks = []
    for sample_idx, sample in enumerate(samples):
        if "unique_sg" in sample:
            continue  # Skip samples that already have parsed scene graphs
        
        conversation = sample.get("conversations", [])
        for conv_idx, turn in enumerate(conversation):
            vlm_response = turn.get("response", "")
            # response_list = [q for q in vlm_response.strip('\n').split('\n') if len(q)]
            response_list = [vlm_response.strip('\n')]
            for question_idx, question in enumerate(response_list):
                parsing_tasks.append((sample_idx, conv_idx, question_idx, question))
    
    print(f"Found {len(parsing_tasks)} questions to parse across {len(samples)} samples")
    
    # Parse all questions in parallel
    results = {}  # (sample_idx, conv_idx, question_idx) -> refined_sg
    
    def parse_single_task(idx_tuple):
        """Wrapper to parse a single task"""
        sample_idx, conv_idx, question_idx, question = idx_tuple
        try:
            refined_sg = parse_single_question(agent, question)
            return (sample_idx, conv_idx, question_idx), refined_sg
        except Exception as e:
            print(f"Error parsing task ({sample_idx}, {conv_idx}, {question_idx}): {e}")
            return (sample_idx, conv_idx, question_idx), []
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(parse_single_task, task) for task in parsing_tasks]
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Parsing scene graphs"):
            key, refined_sg = future.result()
            results[key] = refined_sg
    
    # Group results by sample and conversation
    sample_results = {}  # sample_idx -> (sg_dict, unique_sg_set)
    
    for sample_idx, sample in enumerate(samples):
        if "unique_sg" in sample:
            continue  # Skip samples that already have parsed scene graphs
        
        conversation = sample.get("conversations", [])
        output_dict = {}
        unique_sg = []
        
        for conv_idx, turn in enumerate(conversation):
            vlm_response = turn.get("response", "")
            # response_list = [q for q in vlm_response.strip('\n').split('\n') if len(q)]
            response_list = [vlm_response.strip('\n')]
            lsg_list = []
            for question_idx, question in enumerate(response_list):
                key = (sample_idx, conv_idx, question_idx)
                if key in results:
                    refined_sg = results[key]
                    lsg_list += refined_sg
                    unique_sg += [f"( {' , '.join(sg)} )" for sg in refined_sg]
            
            if lsg_list:
                output_dict[conv_idx] = lsg_list
        
        sample_results[sample_idx] = (output_dict, set(unique_sg))
    
    return sample_results


def batch_compute_distances(samples, timeout, max_workers=10, progress_callback=None, method="delta_con"):
    """
    Batch compute graph distances for multiple samples in parallel.
    
    Args:
        samples: List of samples to process
        timeout: Timeout for graph edit distance calculation
        max_workers: Maximum number of parallel threads
        progress_callback: Optional callback(sample_idx, dist_score)
        method: Graph distance method (e.g. 'delta_con', 'ged', 'netlsd', ...)
    
    Returns:
        Dictionary mapping sample_idx -> dist_score
    """
    # Prepare all distance computation tasks
    distance_tasks = []
    for sample_idx, sample in enumerate(samples):

        if "sg" in sample.keys():
            relationships = sample["sg"]["relationships"]
            objects = sample["sg"]["objects"]
            attributes = sample["sg"]["objects"]
        else:
            relationships = sample["relationships"]
            objects = sample["objects"]
            attributes = sample["attributes"]

        gt_graph = scene_graph_to_nx(relationships, attributes, objects)

        if "unique_sg" in sample and sample["unique_sg"]:
            pred_graph = build_graph_from_string(' , '.join(sample["unique_sg"]))
        else:
            # Skip samples without parsed scene graphs
            continue

        distance_tasks.append((sample_idx, gt_graph, pred_graph, timeout, method))

    
    print(f"Computing distances for {len(distance_tasks)} samples (method={method})...")
    
    results = {}  # sample_idx -> dist_score
    
    def compute_single_distance(task_tuple):
        """Compute distance for a single sample"""
        sample_idx, gt_graph, pred_graph, timeout, method = task_tuple
        # try:
        dist_score = compare_scene_graphs(gt_graph, pred_graph, timeout, method=method)
        return sample_idx, dist_score
        # except Exception as e:
        #     print(f"Error computing distance for sample {sample_idx}: {e}")
        #     return sample_idx, None
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_single_distance, task) for task in distance_tasks]
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Computing graph distances"):
            sample_idx, dist_score = future.result()
            results[sample_idx] = dist_score
            if progress_callback is not None:
                progress_callback(sample_idx, dist_score)
    
    return results


def show_graph(graph, figsize=(18, 14), layout="spring"):
    plt.figure(figsize=figsize)

    # choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=1.5, iterations=100)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "shell":
        pos = nx.shell_layout(graph)
    else:
        pos = nx.spring_layout(graph, k=1.5)

    # draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=2000,
        node_color="skyblue",
        edgecolors="black"
    )

    # draw edges with curve
    nx.draw_networkx_edges(
        graph, pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=50,
        width=2,
        edge_color="gray",
        connectionstyle="arc3,rad=0.2",
    min_target_margin = 20,  # push arrowhead out from target node
    min_source_margin = 15
    )

    # node labels
    labels = {n: data.get("name", n) for n, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(
        graph, pos,
        labels,
        font_size=14,
        font_weight="bold"
    )

    # edge labels
    edge_labels = {
        (u, v): ",".join(data.get("predicates", []))
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_size=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/caption/Ovis2-8B.json")
    parser.add_argument('--outdir', type=str, default="output/caption/sg")
    parser.add_argument('--output_file', type=str, default="Ovis2-8B_sg.json")
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel threads for batch parsing')
    parser.add_argument('--distance_workers', type=int, default=10, help='Maximum number of parallel threads for distance computation')
    parser.add_argument('--merge_first', action='store_true', help='先按 image_id 合并多条记录再算 distance（与 merge_conversations 逻辑一致）')
    parser.add_argument('--distance_only', action='store_true', help='仅计算 distance，不跑 LLM 解析；输入文件需已含 unique_sg')
    parser.add_argument('--method', type=str, default='delta_con',
                        choices=['delta_con', 'ged', 'netlsd', 'portrait', 'hamming', 'him', 'frobenius', 'graph_diffusion', 'jaccard', 'netsimile', 'resistance'],
                        help='Graph distance method (default: delta_con)')
    parser.add_argument('--flush_interval', type=int, default=0, help='每处理 N 个样本就将当前结果写回一次 (0 表示只在最后写一次)')
    parser.add_argument('--random_sample', action='store_true', help='Randomly shuffle samples before selecting sample_num (for consistency testing across runs)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for --random_sample (default: None = different each run)')
    parser.add_argument('--llm_model', type=str, default='gpt-5', help='LLM model name passed to LLMChat for SG parsing (e.g. Qwen3-30B-A3B-Instruct-2507)')
    args = parser.parse_args()

    samples = json.load(open(args.conv_script, "r"))

    if args.merge_first:
        n_before = len(samples)
        samples = merge_conversations_by_image_id(samples)
        print(f"已按 image_id 合并: {n_before} 条 -> {len(samples)} 条")

    if args.random_sample:
        import random
        rng = random.Random(args.random_seed)
        samples = rng.sample(samples, min(args.sample_num, len(samples)))
        print(f"Random sampling: drew {len(samples)} samples from {len(json.load(open(args.conv_script)))} total")
        samples_to_process = samples
    else:
        samples_to_process = samples[:args.sample_num]

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, args.output_file)

    # If an output file already exists, merge in any already-parsed unique_sg
    if os.path.exists(output_path):
        try:
            existing = json.load(open(output_path, "r"))
            existing_by_id = {s.get("image_id"): s for s in existing if s.get("image_id") is not None}
            for s in samples_to_process:
                cached = existing_by_id.get(s.get("image_id"))
                if cached and cached.get("unique_sg") and not s.get("unique_sg"):
                    s["unique_sg"] = cached["unique_sg"]
                if cached and cached.get("parsed_sg") and not s.get("parsed_sg"):
                    s["parsed_sg"] = cached["parsed_sg"]
            print(f"Step 0: Loaded cached parsed results from {output_path}")
        except Exception:
            pass

    # First pass: batch parse（仅在非 distance_only 且样本缺 unique_sg 时执行）
    if not args.distance_only:
        need_parse = not (samples_to_process and any(s.get("unique_sg") for s in samples_to_process))
        if need_parse:
            print("Step 1: Batch parsing scene graphs...")
            agent = LLMChat(args.llm_model)
            batch_results = batch_parse_scene_graph(samples_to_process, agent, max_workers=args.max_workers)
            for sample_idx, (sg_dict, unique_sg) in batch_results.items():
                samples_to_process[sample_idx]["unique_sg"] = list(unique_sg)
                samples_to_process[sample_idx]["parsed_sg"] = sg_dict
            with open(output_path, "w") as file:
                json.dump(samples_to_process, file, indent=4)
        else:
            print("Step 1: 已有 unique_sg，跳过解析")
    else:
        print("Step 1: 跳过解析（--distance_only）")

    # Second pass: batch compute graph distances
    print("Step 2: Computing graph distances...")
    
    dist_list = []
    flush_state = {"finished": 0}
    
    def progress_callback(sample_idx, dist_score):
        if dist_score is not None:
            samples_to_process[sample_idx]["dist_score"] = dist_score
            dist_list.append(dist_score)
            print(f'Sample {sample_idx + 1} distance', dist_score)
        else:
            print(f'Sample {sample_idx + 1} distance computation failed')
    
        flush_state["finished"] += 1
        if args.flush_interval and flush_state["finished"] % args.flush_interval == 0:
            with open(output_path, "w") as file:
                json.dump(samples_to_process, file, indent=4)
            print(f'Flushed intermediate results for {flush_state["finished"]} samples')
    
    distance_results = batch_compute_distances(
        samples_to_process,
        args.timeout,
        max_workers=args.distance_workers,
        progress_callback=progress_callback,
        method=args.method,
    )
    
    # Fallback: try sequential computation for samples that failed in batch
    for sample_idx, sample in enumerate(samples_to_process):
        if sample_idx in distance_results:
            continue
    
        print(f'Warning: Sample {sample_idx + 1} missing in batch results, computing sequentially...')
        if "sg" in sample.keys():
            relationships = sample["sg"]["relationships"]
            objects = sample["sg"]["objects"]
            attributes = sample["sg"]["objects"]
        else:
            relationships = sample["relationships"]
            objects = sample["objects"]
            attributes = sample["attributes"]
    
        gt_graph = scene_graph_to_nx(relationships, attributes, objects)
    
        if "unique_sg" in sample and sample["unique_sg"]:
            pred_graph = build_graph_from_string(' , '.join(sample["unique_sg"]))
            dist_score = compare_scene_graphs(gt_graph, pred_graph, args.timeout, method=args.method)
            sample["dist_score"] = dist_score
            dist_list.append(dist_score)
            print(f'Sample {sample_idx + 1} distance', dist_score)
        else:
            print(f'Sample {sample_idx + 1} missing parsed scene graph, skipping...')
    
    # Final write after all distances (including fallbacks) are computed
    with open(output_path, "w") as file:
        json.dump(samples_to_process, file, indent=4)
    
    if dist_list:
        print('Average distance', sum(dist_list) / len(dist_list))
    else:
        print('No distance scores computed.')
