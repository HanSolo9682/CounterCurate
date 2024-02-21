import json
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

want_test_split = False
folders = ["left_right", "up_down", "counting"]
input_folder = "./train_data"  # Note this must be the absolute folder or the relative folder from your Open-CLIP folder.
output_folder = "."

with open(os.path.join(input_folder, "counting.json"), 'r') as f:
    ct = json.load(f)
with open(os.path.join(input_folder, "left_right.json"), 'r') as f:
    lr = json.load(f)
with open(os.path.join(input_folder, "up_down.json"), 'r') as f:
    ud = json.load(f)


list_of_jsons = []
output_name = ""
for folder in folders:
    if folder == "left_right":
        list_of_jsons.append(lr)
        output_name += folder + "_"
    elif folder == "up_down":
        list_of_jsons.append(ud)
        output_name += folder + "_"
    elif folder == "counting":
        list_of_jsons.append(ct)
        output_name += folder
    else:
        print(f"Unknown folder {folder}, skipping")

captions, images, neg_captions, neg_images = [], [], [], []

for i, curr in enumerate(list_of_jsons):
    folder = folders[i]
    for index, metas in tqdm(curr.items()):
        pos_img = os.path.join(input_folder, folder, "pos", f"{index}.jpg")
        for meta in metas:
            image_id = meta["image_id"]
            neg_img = None
            for potential_file in [os.path.join(input_folder, folder, "neg", f"{index}_{image_id}.png"),
                                   os.path.join(input_folder, folder, "neg", f"{index}_{image_id}.jpg"),
                                   os.path.join(input_folder, folder, "neg", f"{index}.jpg")]:
                if os.path.isfile(potential_file):
                    neg_img = potential_file
                    break
            if neg_img is None: continue
            captions.append(meta["pos"])
            neg_captions.append(meta["neg"])
            images.append(pos_img)
            neg_images.append(neg_img)

if want_test_split:
    captions_train, captions_test, images_train, images_test, neg_captions_train, neg_captions_test, neg_images_train, neg_images_test = train_test_split(captions, images, neg_captions, neg_images, test_size=0.2)
else:
    captions_train, images_train, neg_captions_train, neg_images_train = captions, images, neg_captions, neg_images

df = pd.DataFrame({"captions": captions_train, "images": images_train, "neg_captions": neg_captions_train, "neg_images": neg_images_train})
df.to_csv(os.path.join(output_folder, f"train_{output_name}.csv"), sep="\t", index=False)
if want_test_split:
    df = pd.DataFrame({"captions": captions_test, "images": images_test, "neg_captions": neg_captions_test, "neg_images": neg_images_test})
    df.to_csv(os.path.join(output_folder, f"test_{output_name}.csv"), sep="\t", index=False)