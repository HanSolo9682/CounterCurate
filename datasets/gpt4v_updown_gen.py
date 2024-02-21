import requests
import base64
import json
import os
import openai
import time
import json
from tqdm import tqdm
import threading


NUM_SECONDS_TO_SLEEP = 5
MAX_NUM_TRIALS = 20
output_folder = "."


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

temperature= 0
output_file = os.path.join(output_folder, 'gpt4v_updown_gen.json')

try:
    with open(output_file, "r") as f:
        save_data = json.load(f)
    visited = set(item["idx"] for item in save_data)
except:
    save_data = []
    visited = set()

ans_file = output_file


def thread_func(index, image_id, neg, client_index):

    trial_num = 0
    get_res = True
    question = """I will give you a caption in the format "A is above B."
    you need to expand the sentence such that the meaning "A is above B" is preserved
    and your answer is reasonable for a human to understand what you're describing. 
    Do not make the answer too long; one long sentence is enough. 
    For example, if i give you "a man is under a dog", a good answer would be 
    "there is a man resting on the ground, and there is a dog lying above him." 
    One restriction: A and B do not overlap. This means that if I ask you to expand "a hat is below water", 
    you must not assume that the hat is below water. 
    Remember that you MUST include both A and B in your answer, like my example did. 
    """
    prompt = f"Now, given the caption \"{neg}\", your answer:"
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
                        question + "\n" + prompt
                    ]  
                }
            ],
            # "detail": image_edit_detail,
            "max_tokens": 1024, 
            "stream": False 
            } 
            
            GPT4V_KEY_ENDPOINT = ALL_GPT4V_KEY_ENDPOINT[client_index%len(ALL_GPT4V_KEY_ENDPOINT)]
            #client_index+=1
            GPT4V_KEY, GPT4_V_ENDPOINT = GPT4V_KEY_ENDPOINT['GPT4V_KEY'], GPT4V_KEY_ENDPOINT['GPT4_V_ENDPOINT']
            headers = {
                "Content-Type": "application/json",
                "api-key": GPT4V_KEY,
                # "detail": image_edit_detail,
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
            idx=index+"_"+image_id,
            query=prompt,
            conv=response.json()
    ))

with open("./up_down.json", 'r') as f:
    up_down = json.load(f)
thread_idx = 0
for index, metas in tqdm(up_down.items()):
    for meta in metas:
        if index + "_" + str(meta["image_id"]) in visited: continue
        while threads[thread_idx] is not None:
            tqdm.write(f"waiting for thread {thread_idx} to finish.")
            threads[thread_idx].join(5)
            if not threads[thread_idx].is_alive(): break
            thread_idx = (thread_idx + 1) % LEN_ENDPOINTS
        threads[thread_idx] = threading.Thread(target=thread_func, args=(index, str(meta["image_id"]), meta["neg"], thread_idx))
        threads[thread_idx].start()
        thread_idx = (thread_idx + 1) % LEN_ENDPOINTS
        if thread_idx == 0:
            with open(ans_file, 'w') as f:
                json.dump(save_data, f, indent=4)

with open(ans_file, 'w') as f:
    json.dump(save_data, f, indent=4)