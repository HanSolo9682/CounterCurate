from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import json
import sys
import subprocess


flickr30k_images_path = "./flickr30k-images"
flickr30k_entities_path = "./flickr30k_entities"
marked_images_path = "./marked_flickr30k_images"

if not os.path.exists(marked_images_path):
    os.mkdir(marked_images_path)

sys.path.append(flickr30k_entities_path)
from flickr30k_entities_utils import get_sentence_data, get_annotations
if not os.path.exists(os.path.join(flickr30k_entities_path, "Sentences")):
    subprocess.call(["unzip", os.path.join(flickr30k_entities_path, "annotations.zip"), "-d", flickr30k_entities_path])

files = os.listdir(flickr30k_images_path)
all_colors = {}
for file in tqdm(files):
    index = file.split(".")[0]
    if not index.isdigit(): continue

    sentences = get_sentence_data(os.path.join(flickr30k_entities_path, "Sentences", f"{index}.txt"))
    annotations = get_annotations(os.path.join(flickr30k_entities_path, "Annotations", f"{index}.xml"))
    with Image.open(os.path.join(flickr30k_images_path, f"{index}.jpg")) as img:
        w, h = img.size

        # Define colors as a dictionary with color names and their RGB triplets
        colors = [
            ("Red", (255, 0, 0)),
            ("Green", (0, 255, 0)),
            ("Blue", (0, 0, 255)),
            ("Yellow", (255, 255, 0)),
            ("Magenta", (255, 0, 255)),
            ("Cyan", (0, 255, 255)),
            ("Maroon", (128, 0, 0)),
            ("Olive", (128, 128, 0)),
            ("Dark Green", (0, 128, 0)),
            ("Purple", (128, 0, 128))
        ]

        visited = {}
        colors_per_sentence = {}
        for i in range(len(sentences)):
            curr_colors = {}
            boxes = annotations["boxes"]
            color_index = 0
            # Create an empty image with white background
            res = Image.new('RGB', (w, h), color=(255, 255, 255))
            res.paste(img, (0, 0))
            draw = ImageDraw.Draw(res)
            for phrase in sentences[i]["phrases"]:
                box = phrase["phrase_id"]
                if box in annotations["nobox"] or box in annotations["scene"] or box == '0': continue
                coords = boxes[box]
                curr_colors[box] = []
                for coord in coords:
                    if not visited.keys().__contains__(tuple(coord)): 
                        x1, y1, x2, y2 = coord
                        draw.rectangle([x1, y1, x2, y2], outline=colors[color_index][1])
                        visited[tuple(coord)] = colors[color_index][0]
                        if color_index < 9:
                            color_index += 1
                    curr_colors[box].append(visited[tuple(coord)])
                

            new_image_path = os.path.join(marked_images_path, f'{index}_{i}.jpg')
            res.save(new_image_path)
            colors_per_sentence[i] = curr_colors
    all_colors[index] = colors_per_sentence

with open(os.path.join(marked_images_path, "marked_flickr30k_annotations.json"), "w") as f:
    json.dump(all_colors, f, indent=4)