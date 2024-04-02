import json
import open_clip
from PIL import Image
from tqdm import tqdm
import torch
import os

output_path = '.'

with open("../pointingqa/Datasets/LookTwiceQA/looktwiceqa_dataset.json", 'r') as f:
    ds = json.load(f)
    
with open("./vg_dict.json", 'r') as f:
    vg_dict = json.load(f)

tokenizer = open_clip.get_tokenizer('ViT-B-32')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="laion2b_s34b_b79k")
def test(model, preprocess):
    correct = 0
    count = 0
    results = {}
    with torch.no_grad(), torch.cuda.amp.autocast():
        for index, items in tqdm(vg_dict.items()):
            with Image.open(f"./visual_genome/{index}.jpg") as img:
                image_features = model.encode_image(preprocess(img).unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for i, item in enumerate(items):
                name, ans = item['name'], item['ans']
                text_features = model.encode_text(tokenizer([f"there are {ans} {name} in the image.", f"there are {int(ans)+1} {name} in the image."]))
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
                correct1 = text_probs[0] > text_probs[1]
                correct += correct1
                results[i] = (f"there are {ans} {name} in the image.", f"there are {int(ans)+1} {name} in the image.", f"../../vg_compressed/{index}.jpg", correct1.item())
                count += 1
        
    print(correct / count)
    with open(os.path.join(output_path, "clip_counting_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

test(model, preprocess)