from openai import AzureOpenAI
import os
import requests
import json
from tqdm import tqdm
import threading
from PIL import Image

input_folder = "./train_data"
output_folder = "."

ALL_DALLE_ENDPOINTS = [
    dict(endpoint="YOUR ENDPOINT",
    key="YOUR KEY"),
]

clients = [
    AzureOpenAI(
        api_version="2023-12-01-preview",  
        api_key=dalle_endpoint["key"],  
        azure_endpoint=dalle_endpoint["endpoint"]
    ) for dalle_endpoint in ALL_DALLE_ENDPOINTS
]

threads: list[threading.Thread] = [None] * len(clients)

with open(os.path.join(input_folder, "dalle_gen_counterfactuals.json"), 'r') as f:
    metas = json.load(f)

# Set the directory for the stored image
image_dir = os.path.join(output_folder, 'dalle_gen_counterfactuals')

# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
    generated = set()
else:
    generated = set(os.listdir(image_dir))

def thread_func(index, client_index, meta):
    for change in ["noun", "adjective", "reverse"]:
        if f"{index}_{change}.jpg" in generated or meta[change] == None: continue
        try:
            result = clients[client_index % len(clients)].images.generate(
                model="dall-e-3" if client_index % 3 == 0 else "dalle-3", # the name of your DALL-E 3 deployment
                prompt=meta[change],
                n=1,
                style="natural",
                quality="hd",
                #size="1792x1024"
            )

            json_response = json.loads(result.model_dump_json())
        except Exception as e:
            tqdm.write(index)
            tqdm.write(str(e))
            continue

        # Initialize the image path (note the filetype should be png)
        image_path = 'temp.png'

        # Retrieve the generated image
        image_url = json_response["data"][0]["url"]  # extract image URL from response
        generated_image = requests.get(image_url).content  # download the image
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)

        with Image.open(image_path) as img:
            compressed_image_path = os.path.join(image_dir, f'{index}_{change}.jpg')
            img.save(compressed_image_path, quality=50)
        


thread_idx = 0
for index, meta in tqdm(metas.items()):
    while threads[thread_idx] is not None:
        tqdm.write(f"waiting for thread {thread_idx} to finish.")
        threads[thread_idx].join(5)
        if not threads[thread_idx].is_alive():
            tqdm.write(f"thread {thread_idx} finished.")
            break
        tqdm.write(f"thread {thread_idx} timeout, next thread")
        thread_idx = (thread_idx + 1) % len(clients)
    threads[thread_idx] = threading.Thread(target=thread_func, args=(index, thread_idx, meta))
    threads[thread_idx].start()
    thread_idx = (thread_idx + 1) % len(clients)