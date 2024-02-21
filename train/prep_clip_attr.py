import json
import pandas as pd
from tqdm import tqdm
import os


input_folder = "./train_data"  # Note this must be the absolute folder or the relative folder from your Open-CLIP folder.
output_folder = "."

with open(os.path.join(input_folder, "flickr30k_image_captions.json"), 'r') as f:
    orig = json.load(f)
with open(os.path.join(input_folder, "dalle_gen_counterfactuals.json"), 'r') as f:
    attr = json.load(f)


outputs_dict = {}
for index, changes in tqdm(attr.items()):
    for change, neg_prompt in changes.items():
        if neg_prompt is None: continue
        if os.path.isfile(os.path.join(input_folder, "dalle_gen_counterfactuals", f"{index}_{change}.jpg")):
            if index not in outputs_dict.keys():
                outputs_dict[index] = dict(caption=orig[index][0], image=os.path.join(input_folder, "flickr30k-images", f"{index}.jpg"))
            outputs_dict[index][f"{change}_caption"]=neg_prompt
            outputs_dict[index][f"{change}_image"]=os.path.join(input_folder, "dalle_gen_counterfactuals", f"{index}_{change}.jpg")

captions, images, noun_captions, noun_images, adj_captions, adj_images, rev_captions, rev_images = [], [], [], [], [], [], [], []
for index in tqdm(outputs_dict.keys()):
    captions.append(outputs_dict[index]["caption"])
    images.append(outputs_dict[index]["image"])
    if "noun_caption" not in outputs_dict[index].keys():
        noun_captions.append("nonexistent")
        noun_images.append("nonexistent")
    else:
        noun_captions.append(outputs_dict[index]["noun_caption"])
        noun_images.append(outputs_dict[index]["noun_image"])
    if "adjective_caption" not in outputs_dict[index].keys():
        adj_captions.append("nonexistent")
        adj_images.append("nonexistent")
    else:
        adj_captions.append(outputs_dict[index]["adjective_caption"])
        adj_images.append(outputs_dict[index]["adjective_image"])
    if "reverse_caption" not in outputs_dict[index].keys():
        rev_captions.append("nonexistent")
        rev_images.append("nonexistent")
    else:
        rev_captions.append(outputs_dict[index]["reverse_caption"])
        rev_images.append(outputs_dict[index]["reverse_image"])


df = pd.DataFrame({"captions": captions, "images": images, "noun_captions": noun_captions, "noun_images": noun_images, "adj_captions": adj_captions, "adj_images": adj_images, "rev_captions": rev_captions, "rev_images": rev_images})
df.to_csv(os.path.join(output_folder, "train_counterfactuals.csv"), sep="\t", index=False)