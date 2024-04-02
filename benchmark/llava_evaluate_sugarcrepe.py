import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import random

from PIL import Image

model_path = "PATH_TO_YOUR_LLAVA_MODEL"
ans_file_path = "LLAVA_ANSWERS.json"

disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
image_folder = "../datasets/val2017"

qs_types = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
questions = []

for qs_type in qs_types:
    with open(f"../sugar-crepe/data/{qs_type}.json", 'r') as f:
        curr_json = json.load(f)
    for id, item in curr_json.items():
        pos = item["caption"]
        neg = item["negative_caption"]
        seq = random.randint(0, 1)
        if seq == 0:
            ab_str = f"A. {pos}\nB. {neg}\n"
            ans = "A"
        else:
            ab_str = f"A. {neg}\nB. {pos}\n"
            ans = "B"
        questions.append(dict(
            qs_type=qs_type,
            question_id=id,
            image=item["filename"],
            ans=ans,
            text=f"<image>\nWhich of the following captions best descibe the image?\n{ab_str}Answer with the option's letter from the given choices directly."
        ))

ans_file = open(ans_file_path, "w")
for line in tqdm(questions):
    idx = line["question_id"]
    image_file = line["image"]
    qs = line["text"]
    cur_prompt = qs
    ans = line["ans"]
    # if model.config.mm_use_im_start_end:
    #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    # else:
    #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            # do_sample=True if temperature > 0 else False,
            # temperature=temperature,
            # top_p=top_p,
            # num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"Q: {cur_prompt}\nA: {outputs}\nGround truth: {ans}")

    ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "text": outputs,
                                "model_id": model_name,
                                "qs_type": line["qs_type"],
                                "ans": ans,
                                "metadata": {}}) + "\n")
    ans_file.flush()
ans_file.close()


answers = open(ans_file_path, "r")

correct = {}
total = {}

all_correct = 0
all_total = 0

for line in answers:
    obj = json.loads(line)
    if obj["qs_type"] not in correct.keys():
        correct[obj["qs_type"]] = 0
        total[obj["qs_type"]] = 0
    correct[obj["qs_type"]] += obj["ans"] in obj["text"]
    total[obj["qs_type"]] += 1
    all_correct += obj["ans"] in obj["text"]
    all_total += 1

answers.close()

for qs in correct.keys():
    mean = correct[qs] / total[qs]
    print(f"{qs}: {mean}")
mean = all_correct / all_total
print(f"total: {mean}")
