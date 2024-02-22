import json
import os
import sys
import subprocess


flickr30k_entities_path = "./flickr30k_entities"
output_folder = "."


sys.path.append(flickr30k_entities_path)
from flickr30k_entities_utils import get_sentence_data
if not os.path.exists(os.path.join(flickr30k_entities_path, "Sentences")):
    subprocess.call(["unzip", os.path.join(flickr30k_entities_path, "annotations.zip"), "-d", flickr30k_entities_path])

output = {}

for filename in os.listdir(os.path.join(flickr30k_entities_path, "Sentences")):
    index = filename.split(".")[0]
    sentences = get_sentence_data(os.path.join(flickr30k_entities_path, "Sentences", f"{filename}"))
    output[index] = [sentence["sentence"] for sentence in sentences]

with open(os.path.join(output_folder, "flickr30k_image_captions.json"), 'w') as f:
    json.dump(output, f, indent=4)