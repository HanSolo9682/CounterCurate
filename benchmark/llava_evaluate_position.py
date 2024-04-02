import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import random

model_path = "PATH_TO_YOUR_LLAVA_MODEL"
ans_file_path = "LLAVA_ANSWERS.json"
image_folder = "../train_data"
test_data_path = "./test_pos.json"

disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

questions = []
with open(test_data_path, 'r') as f:
    test_pos = json.load(f)
for item in test_pos:
    questions.append(dict(
        question_id=item["id"],
        image=item["image"],
        text=item["conversations"][0]["value"],
        ans=item["conversations"][1]["value"]
    )) 
random.shuffle(questions)
ans_file = open(ans_file_path, "w")
for line in tqdm(questions):
    idx = line["question_id"]
    image_file = line["image"]
    qs = line["text"]
    cur_prompt = qs
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
    truth = line["ans"]
    print(f"Q: {cur_prompt}\nA: {outputs}\nground truth:{truth}")

    ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "text": outputs,
                                "model_id": model_name,
                                "ans": truth,
                                "metadata": {}}) + "\n")
    ans_file.flush()
ans_file.close()