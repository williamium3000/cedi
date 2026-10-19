"""Turn the LLM's scene-graph parse ("<subject, predicate, object>" groups) into triplets,
keeping only physical objects. Used by grader/sg/graph_distance.py."""
import os

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from utils.construct_tree import get_hypernym_tree, get_object_map

_OBJECT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                            "filtered_object_synsets_final.json")
object_dict = get_object_map(object_file_path=_OBJECT_FILE)

filtered_non_phy_words = ["city", "image", "city life", "landscape", "light", "time", "diversity", 'traffic rule', 'cityscape','activity','scene','environment', 'workspace', 'area']


def is_physical_word(subject_word, sg_tuple, object_dict):
    if subject_word.lower() in filtered_non_phy_words:
        return False
    elif subject_word.lower() in object_dict.keys() or subject_word in object_dict.values():
        return True
    else:
        phrase = sg_tuple.replace(',', '').lower()
        best_synset = lesk(phrase, subject_word, pos=wn.NOUN)
        if best_synset:
            syn = wn.synset(best_synset.name())
            hypernym_tree = get_hypernym_tree(syn)
            if any(hypernym.name().startswith(
                    ("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for
                   hypernym in hypernym_tree):
                return True

        for word in subject_word.split(' '):
            best_synset = lesk(phrase, word, pos=wn.NOUN)

            if best_synset:
                syn = wn.synset(best_synset.name())
                hypernym_tree = get_hypernym_tree(syn)
                if any(hypernym.name().startswith(("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for hypernym in hypernym_tree):
                    # print(True, syn)
                    return True
        # print(False, syn)
        return False


def refine_output(response):
    before_match_parsing = []
    if 'none' in response.lower():
        return []

    response = response.replace('>,', '>')
    for a in response.split("<")[1:]:
        a = a.split(",")
        # if len(a) < 3:
        #     continue
        if len(a) == 3 and a[1].strip() not in ['is', 'was', 'are', 'were']:
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            sub_status = is_physical_word(sub, ' '.join(a), object_dict)
            obj_status = is_physical_word(obj, ' '.join(a), object_dict)
            if sub_status and obj_status:
                before_match_parsing.append([sub, pred, obj])
            elif obj_status:
                before_match_parsing.append([obj])
            elif sub_status:
                before_match_parsing.append([sub])

        elif len(a) == 1:
            sub = a[0].split(">")[0].strip()
            if is_physical_word(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub])
        elif len(a) == 3 and a[1].strip() in ['is', 'was', 'are', 'were']:
            sub, pred, adj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            if is_physical_word(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub, pred, adj])
        else:
            print('Error format', a)

    return before_match_parsing
