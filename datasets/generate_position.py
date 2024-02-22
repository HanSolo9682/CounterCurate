import os
import json
import re
import sys
import subprocess

flickr30k_path = "./flickr30k-images"
flickr30k_entities_path = "./flickr30k_entities"
output_folder = "."
FIND = "lr"
#FIND = "ud"


sys.path.append(flickr30k_entities_path)
from flickr30k_entities_utils import get_sentence_data, get_annotations
if not os.path.exists(os.path.join(flickr30k_entities_path, "Sentences")):
    subprocess.call(["unzip", os.path.join(flickr30k_entities_path, "annotations.zip"), "-d", flickr30k_entities_path])

if FIND == "lr":
    POS1 = "to the left of"
    POS2 = "to the right of"
    folder = "left_right"
else:
    POS1 = "above"
    POS2 = "below"
    folder = "up_down"

def box_relation(box1, box2):
    '''Returns -1 if box1 is to the left/above of box2, 1 if box1 is to the right/below of box2, and 0 otherwise (overlap).'''
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    if FIND == "lr":
        if ax2 <= bx1:
            return -1
        elif ax1 >= bx2:
            return 1
        else:
            return 0
    else:
        if ay2 <= by1:
            return -1
        elif ay1 >= by2:
            return 1
        else:
            return 0
    
def id_to_phrase(sentences):
    '''format: {id: (phrase, [list of appearance])}'''
    result = {}
    for i, sentence in enumerate(sentences):
        for phrase in sentence["phrases"]:
            if phrase["phrase_id"] not in result.keys():
                result[phrase["phrase_id"]] = (phrase["phrase"], [i])
            else:
                result[phrase["phrase_id"]][1].append(i)
    return result

def overlapping_sentence(id1, id2, id_phrase_dict):
    if id1 not in id_phrase_dict.keys() or id2 not in id_phrase_dict.keys():
        return -1
    l1 = id_phrase_dict[id1][1]
    l2 = id_phrase_dict[id2][1]
    for a in l1:
        for b in l2:
            if a == b:
                return a
    return -1


def filter(outputs, type):
    pattern = re.compile(r"^(.+) is to the (.+) of (.+)$") if type == "lr" \
         else re.compile(r"^(.+) is (above|below) (.+)$")
    filter_list = set(['someone', 'others', 'the', 'one', 'other', \
                       'left', 'right', 'stuff', 'a', 'something', 'thing'])

    indexes_to_remove = set()

    for index in outputs.keys():
        for i in range(len(outputs[index]) - 1, -1, -1):
            pos: str = outputs[index][i]["pos"]
            if "another" in pos or "other" in pos:
                indexes_to_remove.add(index)
                break

            pos = pos.lower()
            word1, _, word2 = re.findall(pattern, pos)[0]
            word1 = word1.replace("one ", "").replace("another ", "").replace("the ", "").replace("a ", "").replace("other ", "")
            word2 = word2.replace("one ", "").replace("another ", "").replace("the ", "").replace("a ", "").replace("other ", "")
            if word1 == word2 or word1 in filter_list or word2 in filter_list:
                indexes_to_remove.add(index)
                break

    for index in indexes_to_remove:
        del outputs[index]

    return outputs


results = {}

image_files = os.listdir(flickr30k_path)
for image_file in image_files:
    index = image_file.split('.')[0]
    if not index.isnumeric(): continue
    sentences = get_sentence_data(os.path.join(flickr30k_entities_path, "Sentences", f"{index}.txt"))
    annotations = get_annotations(os.path.join(flickr30k_entities_path, "Annotations", f"{index}.xml"))
    boxes = annotations["boxes"]
    if len(boxes) < 2: continue
    id_phrase_dict = id_to_phrase(sentences)
    results[index] = []

    visited_phrase_pairs = set()
    image_id = 0
    for phrase1_id, boxes1 in boxes.items():
        for phrase2_id, boxes2 in boxes.items():
            if phrase1_id == phrase2_id or len(boxes1) != 1 or len(boxes2) != 1 \
            or visited_phrase_pairs.__contains__((phrase1_id, phrase2_id)): continue

            relation = box_relation(boxes1[0], boxes2[0])
            if relation == 0: continue

            sentence_id = overlapping_sentence(phrase1_id, phrase2_id, id_phrase_dict)
            if sentence_id == -1: continue

            phrase1, phrase2 = id_phrase_dict[phrase1_id][0], id_phrase_dict[phrase2_id][0]
            if phrase1 == phrase2: continue
            visited_phrase_pairs.add((phrase1_id, phrase2_id))
            visited_phrase_pairs.add((phrase2_id, phrase1_id))

            position = POS1 if relation == -1 else POS2
            reverse_position = POS2 if position == POS1 else POS1
            coords = [boxes[phrase1_id][0], boxes[phrase2_id][0]]

            # coords & phrases are only needed by GLIGEN to generate above-below negatives
            results[index].append(
                dict(
                    image_id=image_id,
                    phrases=[phrase1, phrase2],
                    coords=coords,
                    pos=f"{phrase1} is {position} {phrase2}",
                    neg=f"{phrase1} is {reverse_position} {phrase2}"
                )
            )
            image_id += 1
    
    if len(results[index]) == 0:
        del results[index]

with open(os.path.join(output_folder, f"{folder}.json"), 'w') as f:
    json.dump(filter(results, FIND), f, indent=4)
