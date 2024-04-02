import sys
sys.path.append("./GLIGEN")

from gligen_inference import run, load_ckpt
import torch
import argparse
import json
from PIL import Image
import re
from tqdm import tqdm
    
output_path = "."
checkpoint_path = "./ckpts"


outputs_dict = {}
with open("./gpt4v_updown_gen.json", 'r') as f:
    responses = json.load(f)

shared_meta = dict(
    ckpt=os.path.join(checkpoint_path, "gligen-generation-text-box", "diffusion_pytorch_model.bin"),
    save_folder_name="infer_up_down"
)

expanded_prompts = dict()
for response in responses:
    idx, conv = response["idx"], response["conv"]
    try:
        expanded_prompts[idx] = conv["choices"][0]["message"]["content"]
    except KeyError:
        continue


def swap_locations(locations):
    ''' Given a list of two boxes of two objects, swap their relative location but preserving original box size '''
    # Ensure the boxes are represented as (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = locations[0]
    x1_2, y1_2, x2_2, y2_2 = locations[1]

    # Calculate the width and height of the boxes
    width_1 = x2_1 - x1_1
    height_1 = y2_1 - y1_1
    xcenter_1 = (x1_1 + x2_1) / 2
    ycenter_1 = (y1_1 + y2_1) / 2

    width_2 = x2_2 - x1_2
    height_2 = y2_2 - y1_2
    xcenter_2 = (x1_2 + x2_2) / 2
    ycenter_2 = (y1_2 + y2_2) / 2

    # Swap the x-coordinates and y-coordinates while keeping the width and height constant
    new_x1_1 = max(0, xcenter_2 - width_1 / 2)
    new_x2_1 = min(1, xcenter_2 + width_1 / 2)
    new_x1_2 = max(0, xcenter_1 - width_2 / 2)
    new_x2_2 = min(1, xcenter_1 + width_2 / 2)

    new_y1_1 = max(0, ycenter_2 - height_1 / 2)
    new_y2_1 = min(1, ycenter_2 + height_1 / 2)
    new_y1_2 = max(0, ycenter_1 - height_2 / 2)
    new_y2_2 = min(1, ycenter_1 + height_2 / 2)

    # Return the swapped bounding boxes
    swapped_box1 = (new_x1_1, new_y1_1, new_x2_1, new_y2_1)
    swapped_box2 = (new_x1_2, new_y1_2, new_x2_2, new_y2_2)

    return [swapped_box1, swapped_box2]

model, autoencoder, text_encoder, diffusion, config = load_ckpt(shared_meta["ckpt"])

for index in tqdm(results.keys()):
    with Image.open(f"./flickr30k-images/{index}.jpg") as img:
        w, h = img.size
    for package in results[index]:
        full_index = index + "_" + str(package["image_id"])
        if full_index in outputs_dict.keys() or full_index not in expanded_prompts.keys():
            continue
        phrases = package["phrases"]
        locations = []
        for coord in package["coords"]:
            locations.append([coord[0] / w, coord[1] / h, coord[2] / w, coord[3] / h])
        locations = swap_locations(locations)
        meta = dict(
            prompt=expanded_prompts[full_index],
            phrases=phrases,
            locations=locations
        ) | shared_meta

        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", type=str,  default="./", help="root folder for output")


        parser.add_argument("--batch_size", type=int, default=1, help="")
        parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
        parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
        parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
        #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
        args = parser.parse_args()

        starting_noise = torch.randn(args.batch_size, 4, 64, 64).to("cuda")
        starting_noise = None

        image_id = run(meta, args, model, autoencoder, text_encoder, diffusion, config, starting_noise)[0]
        outputs_dict[index + "_" + str(package["image_id"])] = image_id


with open(os.path.join(output_path, "outputs_dict_up_down.json"), 'w') as f:
    json.dump(outputs_dict, f, indent=4)