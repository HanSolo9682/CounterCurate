import sys
sys.path.append("./GLIGEN")

from gligen_inference import run, load_ckpt
import argparse
import json
from PIL import Image
from tqdm import tqdm
import os
    
output_path = "."
checkpoint_path = "./ckpts"


outputs_dict = {}
shared_meta = dict(
    ckpt=os.path.join(checkpoint_path, "gligen-inpainting-text-image-box", "checkpoint_inpainting_text_image.bin"),
    save_folder_name="infer_count"
)

model, autoencoder, text_encoder, diffusion, config = load_ckpt(shared_meta["ckpt"])


def gen(index, meta):
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="./", help="root folder for output")

    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()

    # starting_noise = torch.randn(args.batch_size, 4, 64, 64).to("cuda")
    starting_noise = None

    image_id = run(meta, args, model, autoencoder, text_encoder, diffusion, config, starting_noise)[0]
    outputs_dict[index] = image_id


for index, metas in tqdm(results.items()):
    for i, meta in enumerate(metas):
        meta |= shared_meta
        if f"{index}_{i}" in outputs_dict.keys(): continue
        with Image.open(f"./flickr30k-images/{index}.jpg") as img:
            w, h = img.size
        locations = []
        for location in meta["locations"]:
            locations.append([location[0] / w, location[1] / h, location[2] / w, location[3] / h])
        meta["locations"] = locations
        gen(index + "_" + str(i), meta)


with open(os.path.join(output_path, "outputs_dict_counting.json"), 'w') as f:
    json.dump(outputs_dict, f, indent=4)