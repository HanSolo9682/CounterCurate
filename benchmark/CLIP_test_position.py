import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import open_clip

test_data_path = "./Flickr30k-Counterfactuals/clip_sample_data/test_left_right_.csv"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer('ViT-B-32')
df = pd.read_csv(test_data_path, sep="\t")
texts = df["captions"].to_list()
images = df["images"].to_list()
neg_texts = df["neg_captions"].to_list()
neg_images = df["neg_images"].to_list()

def test(model, preprocess):
    model.eval()
    correct = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(len(images))):

            text_features = model.encode_text(tokenizer([texts[i], neg_texts[i]]))
            text_features /= text_features.norm(dim=-1, keepdim=True)

            with Image.open(images[i]) as img:
                image_features = model.encode_image(preprocess(img).unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            correct += text_probs[0] > text_probs[1]

            with Image.open(neg_images[i]) as img:
                image_features = model.encode_image(preprocess(img).unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            correct += text_probs[0] < text_probs[1]
    print(correct / len(images) / 2)

test(model, preprocess)