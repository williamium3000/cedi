from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
import re
import json
import nltk
# nltk.download('punkt_tab')


def is_physical_synset(syn):
    syn = wn.synset(syn)
    hypernym_tree = get_hypernym_tree(syn)
    if any(hypernym.name().startswith(("physical_entity.n.", "people.n.", "vegetation.n")) for hypernym in hypernym_tree):
        # print(True, syn)
        return True
    # print(False, syn)
    return False


def get_object_map(object_file_path='object_synsets.json'):
    with open(object_file_path) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key.split('.')[0]: value for key, value in object_dict.items()}

    filtered_object_dict = {key: value for key, value in object_dict.items() if is_physical_synset(value)}
    return filtered_object_dict


def sentence_to_words(object_dict, sentence):
    object_pattern = r'\b(' + '|'.join(map(re.escape, object_dict.keys())) + r')\b'

    class_list = [' '.join(object_class.split('.')[0].split('_')) for object_class in object_dict.values()]
    class_pattern = r'\b(' + '|'.join(map(re.escape, class_list)) + r')\b'

    class_matches = re.findall(class_pattern, sentence.lower())
    object_matches = re.findall(object_pattern, sentence.lower())

    match_result = {match: object_dict[match] for match in object_matches}
    for match in class_matches:
        best_synset = lesk(sentence.lower().split(), match, pos=wn.NOUN)
        if best_synset and match not in match_result:
            match_result[match] = best_synset.name()
    return match_result


def get_hypernym_tree(synset):
    hypernyms = set()

    def traverse(syn):
        if syn not in hypernyms:
            hypernyms.add(syn)
            for hypernym in syn.hypernyms():
                traverse(hypernym)

    traverse(synset)
    return hypernyms


def get_hyponyms_tree(synset):
    hyponyms = set()

    def traverse(syn):
        if syn not in hyponyms:
            hyponyms.add(syn)
            for hypernym in syn.hyponyms():
                traverse(hypernym)

    traverse(synset)
    return hyponyms


def is_hallucinate(test_word, gt_word):
    if test_word == gt_word:
        return True

    test_synset = wn.synset(test_word)
    gt_synset = wn.synset(gt_word)

    test_hyponyms = get_hyponyms_tree(test_synset)
    gt_hypernyms = get_hypernym_tree(gt_synset)

    return test_synset not in gt_hypernyms or gt_synset not in test_hyponyms


if __name__ == '__main__':
    object_dict = get_object_map(object_file_path='../data/filtered_object_synsets_final.json')
    test_sentence = 'There is a vehicle on the street.'
    gt_synset_list = ['car.n.01']

    test_synset_match_result = sentence_to_words(object_dict, test_sentence)

    for gt_synset in gt_synset_list:
        for test_object, test_object_class in test_synset_match_result.items():
            print(test_object_class, gt_synset, is_hallucinate(test_object_class, gt_synset))

