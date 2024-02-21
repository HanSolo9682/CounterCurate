import os
import json
import random
import sys
import subprocess

flickr30k_path = "./flickr30k-images"
flickr30k_entities_path = "./flickr30k_entities"
output_folder = "."

sys.path.append(flickr30k_entities_path)
from flickr30k_entities_utils import get_sentence_data, get_annotations
if not os.path.exists(os.path.join(flickr30k_entities_path, "Sentences")):
    subprocess.call(["unzip", os.path.join(flickr30k_entities_path, "annotations.zip"), "-d", flickr30k_entities_path])

def non_overlapping_boxes(box1, box2):
    '''Returns 1 if two boxes do not overlap, and 0 otherwise (overlap).'''
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    return ay2 <= by1 or ax2 <= bx1 or ax1 >= bx2 or ay1 >= by2


def id_to_phrase(sentences):
    '''format: {id: (phrase, phrase_type, [list of appearance])}'''
    result = {}
    for i, sentence in enumerate(sentences):
        for phrase in sentence["phrases"]:
            if phrase["phrase_id"] not in result.keys():
                result[phrase["phrase_id"]] = (phrase["phrase"], phrase["phrase_type"][0],[i])
            else:
                result[phrase["phrase_id"]][2].append(i)
    return result


def determine_phrase(id, id_phrase_dict):
    quantifiers = ["1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "several ", "a ", "one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight ", "many ", "group of ", "lot of", "a lot of ", "some ", "other ", "crowd of ", "the ", "bunch of "]
    phr: str = id_phrase_dict[id][0].lower()
    for quantifier in quantifiers:
        if quantifier in phr:
            phr = phr.replace(quantifier, '')
    if phr == "couple":
        phr = "people"
    return phr


def overlapping_sentence(id1, id2, id_phrase_dict):
    if id1 not in id_phrase_dict.keys() or id2 not in id_phrase_dict.keys():
        return -1
    l1 = id_phrase_dict[id1][2]
    l2 = id_phrase_dict[id2][2]
    for a in l1:
        for b in l2:
            if a == b:
                return a
    return -1


image_files = os.listdir(flickr30k_path)
results = {}

for image_file in image_files:
    index = image_file.split('.')[0]
    if not index.isnumeric(): continue
    sentences = get_sentence_data(os.path.join(flickr30k_entities_path, "Sentences", f"{index}.txt"))
    annotations = get_annotations(os.path.join(flickr30k_entities_path, "Annotations", f"{index}.xml"))
    
    all_boxes = annotations["boxes"]
    id_phrase_dict = id_to_phrase(sentences)

    phrases = set()
    repetitive = False
    for id, item in id_phrase_dict.items():
        a = determine_phrase(id, id_phrase_dict)
        if a in phrases: 
            repetitive = True
            break
        phrases.add(a)
    if repetitive: continue
    
    ids = []
    for phrase_id, boxes in all_boxes.items():
        if phrase_id not in id_phrase_dict.keys(): continue
        if id_phrase_dict[phrase_id][1] in ["bodyparts", "scene"]: continue
        if id_phrase_dict[phrase_id][0] in ["all", "some", "others", "group", "the others", "one", "two", "three", "four", "five", "six", "seven", "eight", "the", "many", "1", "2", "3", "4", "5", "6", "7", "8", "a"]: continue
        if len(boxes) >= 2 and len(boxes) <= 8:
            ids.append(phrase_id)
    
    visited_phrase_pairs = set()
    curr = []
    for id1 in ids:
        for id2 in ids:
            if id1 == id2 or (id1, id2) in visited_phrase_pairs: continue
            visited_phrase_pairs.add((id1, id2))
            visited_phrase_pairs.add((id2, id1))
            sentence_id = overlapping_sentence(id1, id2, id_phrase_dict)
            if sentence_id == -1: continue

            boxes1 = all_boxes[id1]
            boxes2 = all_boxes[id2]
            len1 = len(boxes1)
            len2 = len(boxes2)
            overlap_boxes = []
            overlap_obj1_index = -1
            overlap_obj1_count = 1
            overlap_obj2_count = 0
            for i, box1 in enumerate(boxes1):
                for j, box2 in enumerate(boxes2):
                    if not non_overlapping_boxes(box1, box2):
                        if len(overlap_boxes) == 0:
                            overlap_boxes.append(box1)
                            overlap_obj1_index = i
                        overlap_boxes.append(box2)
                        overlap_obj2_count += 1
                        
                if len(overlap_boxes) > 0: break

            if len(overlap_boxes) > 0:
                for i, box1 in enumerate(boxes1):
                    if i == overlap_obj1_index: continue
                    if not non_overlapping_boxes(box1, overlap_boxes[0]):
                        overlap_boxes.insert(1, box1)
                        overlap_obj1_count += 1
            

            phrase1 = determine_phrase(id1, id_phrase_dict)
            phrase2 = determine_phrase(id2, id_phrase_dict)
            if phrase1 == phrase2 or "others" in [phrase1, phrase2]: continue


            prompt = sentences[sentence_id]["sentence"]

            if len1 > len2:  # makes sure that id1 is the phrase that has a fewer or equal count than id2
                id1, id2 = id2, id1
                len1, len2 = len2, len1
                phrase1, phrase2 = phrase2, phrase1
                boxes1, boxes2 = boxes2, boxes1
                overlap_obj1_count, overlap_obj2_count = overlap_obj2_count, overlap_obj1_count
                overlap_boxes.reverse()
            if len(overlap_boxes) > 0:
                new_len1 = len1 - overlap_obj1_count
                new_len2 = len2 - overlap_obj2_count
                if new_len1 == 0 or new_len2 == 0: continue
                locations = overlap_boxes
                phrases = [f"plant no.{i}" for i in range(len(overlap_boxes))]
                prompt = " and ".join([f"plant no.{i}" for i in range(len(overlap_boxes))]) + " thrives in the environment"
            else:
                new_len1 = len1 + 1
                new_len2 = len2 - 1
                locations = [boxes2[random.randint(0, len2-1)]]
                phrases = [phrase1]
                prompt = f"there is a {phrase1}"
            curr.append(dict(
                pos=f"there are {len1} {phrase1} and {len2} {phrase2}.",
                neg=f"there are {new_len1} {phrase1} and {new_len2} {phrase2}.",
                locations=locations,
                phrases=phrases,
                prompt=prompt,
                input_image=os.path.join(flickr30k_path, f"{index}.jpg")
            ))

    if len(curr) != 0:
        results[index] = curr


with open(os.path.join(output_folder, "counting.json"), 'w') as f:
    json.dump(results, f, indent=4)