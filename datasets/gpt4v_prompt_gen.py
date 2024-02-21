import requests
import base64
import json
import os
import openai
import time
import json
from tqdm import tqdm
import threading
import sys
import subprocess


NUM_SECONDS_TO_SLEEP = 5
MAX_NUM_TRIALS = 20
flickr30k_images_path = "./flickr30k-images"
flickr30k_entities_path = "./flickr30k_entities"
marked_images_path = "./marked_flickr30k_images"
output_folder = "."

sys.path.append(flickr30k_entities_path)
from flickr30k_entities_utils import get_sentence_data, get_annotations
if not os.path.exists(os.path.join(flickr30k_entities_path, "Sentences")):
    subprocess.call(["unzip", os.path.join(flickr30k_entities_path, "annotations.zip"), "-d", flickr30k_entities_path])


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

ALL_GPT4V_KEY_ENDPOINT = [
    dict(
        GPT4V_KEY = "YOUR KEY",
        GPT4_V_ENDPOINT = "YOUR ENDPOINT"),
]

LEN_ENDPOINTS = len(ALL_GPT4V_KEY_ENDPOINT)
threads: list[threading.Thread] = [None] * LEN_ENDPOINTS

image_edit_detail = 'high'

temperature = 0
output_file = os.path.join(output_folder, 'gpt4v_neg_prompt.json')
with open(os.path.join(marked_images_path, "marked_flickr30k_annotations.json"), "r") as f:
      colors_file = json.load(f)

with open(output_file, "r") as f:
    save_data = json.load(f)
visited = set(item["idx"] for item in save_data)

ans_file = output_file


def thread_func(index, client_index):
    original_img = encode_image(os.path.join(flickr30k_images_path, f"{index}.jpg"))
    colors = colors_file[index]
    sentences = get_sentence_data(os.path.join(flickr30k_entities_path, "Sentences", f"{index}.txt"))
    annotations = get_annotations(os.path.join(flickr30k_entities_path, "Annotations", f"{index}.xml"))
    with open(os.path.join(flickr30k_entities_path, "Sentences", f"{index}.txt"), 'r') as f:
      enhanced_captions = f.read().split('\n')

    if index + '-0' in visited:
        return
    sentence = sentences[0]
    original_caption = sentence["sentence"]
    phrases = sentence["phrases"]
    marked_img = encode_image(os.path.join(marked_images_path, f"{index}_0.jpg"))
    enhanced_caption = enhanced_captions[0]
    boxes = ""
    for phrase in phrases:
        phrase_id = str(phrase["phrase_id"])
        if phrase_id in annotations["nobox"] or phrase_id in annotations["scene"] or phrase_id == '0': continue
        boxes += "#" + phrase_id + ": " + ", ".join(colors['0'][phrase_id]) + "; "


    trial_num = 0
    get_res = True
    question = """You are given an image, the same image but with bounding boxes, its corresponding caption and an enhanced form of the caption. Their format is as follows:
Original Caption: A child in a pink dress is helping a baby in a blue dress climb up a set of stairs in an entry way.
Enhanced Caption: [/EN#1/people A child] in [/EN#2/clothing a pink dress] helping  [/EN#3/people a baby] in  [/EN#4/clothing a blue dress] climb up [/EN#5/other a set of stairs] in [/EN#6/scene an entry way].
In the enhanced caption, there is no new data, but that each “entity” is marked by a pair of square brackets. Most entities each correspond to one or more bounding boxes, which will be specified. For example, entity 1 in the sentence is “A child”, which is marked by a tag [/EN#1/people …]. “people” states the type of the entity. If entity is “other”, then there are no restrictions applied.
You are tasked to:
Generate a caption that changes the object being discussed in exactly one of the entities. You MUST ensure that the new object is the same type of entity as the original object as specified in the tag. For example: [/EN#1/people A child] => [/EN#1/people An adult] is allowed, but [/EN#1/people A child] => [/EN#1/people A cat] is not allowed because a cat is not “people”;
Generate a caption that changes the qualifier (such as an adjective of a quantifier) that describes the object in exactly one of the entities. For example: [/EN#2/clothing a pink dress] => [/EN#2/clothing a green dress].
Generate, if possible, a caption that reverses two of the entities or their qualifiers such that the original sentence structure is not changed, but produces a negative prompt. For example, given two entities “a green dress” and “a blue blouse”, you can either swap the two entities’ order or swap the adjectives and produce “a blue dress” and “a green blouse”. If you cannot generate one, report None.
                    
All in all, the new description must meet all of these requirements:
1. The change of attribute must be sufficiently different to make the new description inaccurate, but it should also be somewhat related to be challenging to an AI model.
2. Compared to the original description, the new description must differ in only one attribute. All other details must be kept the same.
3. The new description must mimic the sentence structure of the original description.
4. The new description must be fluent, logical, and grammatically correct.
5. Carefully look at the image, and give negative captions that are reasonable given the objects’ position, size, and relationship to the overall setting.
6. Pose challenging(difficult enough) negative captions so that a large multimodal text generation model should struggle to distinguish the original caption v.s. negative caption.

Here are some examples whose output format you should follow:
Original Caption: A child in a pink dress is helping a baby in a blue dress climb up a set of stairs in an entry way.
Enhanced Caption: [/EN#1/people A child] in [/EN#2/clothing a pink blouse] helping  [/EN#3/people a baby] in  [/EN#4/clothing a blue dress] climb up [/EN#5/other a set of stairs] in [/EN#6/scene an entry way].
Bounding Boxes: #1: purple
Your answer:
{“noun”: {“action”: (1, “a child”, “an adult”), “caption”: “An adult in a green dress is helping a baby in a blue dress climb up a set of stairs in an entry way.”], “adjective”: {“action”: (2, “a pink dress”, “a green dress”), “caption”: “A child in a green dress is helping a baby in a blue dress climb up a set of stairs in an entry way.”}, “reverse”: {“action”: (2, 4), “caption”: “A child in a blue blouse is helping a baby in a pink dress climb up a set of stairs in an entry way.”}}
"""
    prompt = f"Original Caption: {original_caption}\nEnhanced Caption: {enhanced_caption}\nBounding boxes: {boxes}\nYour answer:"
    while trial_num<MAX_NUM_TRIALS:
        try:
            get_res = True
            trial_num += 1

            payload = {
            "temperature": temperature,
            "messages": [ 
                {
                    "role": "system", 
                    "content": "You are a helpful assistant." 
                },
                {
                    "role": "user", 
                    "content": [
                        question + "\n" + prompt,
                        {
                            "image": original_img,
                            "detail": image_edit_detail,
                        },
                        {
                            "image": marked_img,
                            "detail": image_edit_detail,
                        }  
                    ]  
                }
            ],
            "max_tokens": 4096, 
            "stream": False 
            } 
            
            GPT4V_KEY_ENDPOINT = ALL_GPT4V_KEY_ENDPOINT[client_index%len(ALL_GPT4V_KEY_ENDPOINT)]
            GPT4V_KEY, GPT4_V_ENDPOINT = GPT4V_KEY_ENDPOINT['GPT4V_KEY'], GPT4V_KEY_ENDPOINT['GPT4_V_ENDPOINT']
            headers = {
                "Content-Type": "application/json",
                "api-key": GPT4V_KEY,
                }
            
            response = requests.post(GPT4_V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()
            if 'error' not in results.keys():
                break
            else:
                print(index, question,  results['error'])
                get_res = False
        except openai.RateLimitError:
            get_res = False
            pass
        except Exception as e:
            get_res = False
            tqdm.write(index + str(e))
        time.sleep(NUM_SECONDS_TO_SLEEP)
        
    if get_res:
        tqdm.write(prompt)
        tqdm.write(index + json.dumps(results))
        tqdm.write("")

    save_data.append(dict(
            idx=index+"-0",
            query=prompt,
            conv=response.json()
    ))

thread_idx = 0
files = os.listdir(flickr30k_images_path)
for file in tqdm(files, leave=False, position=0):
    index = file.split(".")[0]
    if index + "-0" in visited: continue
    while threads[thread_idx] is not None:
        tqdm.write(f"waiting for thread {thread_idx} to finish.")
        threads[thread_idx].join(5)
        if not threads[thread_idx].is_alive(): break
        thread_idx = (thread_idx + 1) % LEN_ENDPOINTS
    threads[thread_idx] = threading.Thread(target=thread_func, args=(index, thread_idx))
    threads[thread_idx].start()
    thread_idx = (thread_idx + 1) % LEN_ENDPOINTS
    if thread_idx == 0:
        with open(ans_file, 'w') as f:
            json.dump(save_data, f, indent=4)

with open(ans_file, 'w') as f:
    json.dump(save_data, f, indent=4)