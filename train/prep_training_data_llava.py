import json
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import random

want_test_split = False
want_attr = False
folders = ["left_right", "up_down", "counting"]
input_folder = "./train_data"  # Note this must be the absolute folder or the relative folder from your LLaVA folder.
output_folder = "."

with open(os.path.join(input_folder, "left_right.json"), 'r') as f:
    lr = json.load(f)
with open(os.path.join(input_folder, "up_down.json"), 'r') as f:
    ud = json.load(f)
with open(os.path.join(input_folder, "counting.json"), 'r') as f:
    ct = json.load(f)

with open(os.path.join(input_folder, "flickr30k_image_captions.json"), 'r') as f:
    orig = json.load(f)
with open(os.path.join(input_folder, "dalle_gen_counterfactuals.json"), 'r') as f:
    attr = json.load(f)

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
        output_name += folder + "_"
    else:
        print(f"Unknown folder {folder}, skipping")

train = []
test = []

questions = []
answers = []
images = []

if want_attr:
    output_name += "counterfactuals"
    texts_dict = {}
    images_dict = {}
    for index, changes in attr.items():
        for change, neg_prompt in changes.items():
            if neg_prompt is None: continue
            if os.path.isfile(os.path.join(input_folder, "dalle_gen_counterfactuals", f"{index}_{change}.jpg")):
                if index not in texts_dict.keys():
                    texts_dict[index] = [orig[index][0]]
                    images_dict[index] = [os.path.join(input_folder, "flickr30k-images", f"{index}.jpg")]
                texts_dict[index].append(neg_prompt)
                images_dict[index].append(os.path.join(input_folder, "dalle_gen_counterfactuals", f"{index}_{change}.jpg"))


    for index in texts_dict.keys():
        texts_temp = texts_dict[index]
        images_temp = images_dict[index]
        curr_question = []
        curr_answers = []
        curr_images = []
        for j in range(len(texts_temp)):
            neg_index = random.randint(0, len(texts_temp) - 1)
            while neg_index == j:
                neg_index = random.randint(0, len(texts_temp) - 1)
            seq = random.randint(0, 1)
            curr_text = texts_temp[j]
            curr_neg_text = texts_temp[neg_index]

            qs = "<image>\nWhich of the following captions best describes the image?\n"
            if seq == 0:
                qs += f"A. {curr_text}\nB. {curr_neg_text}\n"
            else:
                qs += f"A. {curr_neg_text}\nB. {curr_text}\n"
            qs += "Answer with the option's letter from the given choices directly."
            curr_question.append(qs)
            curr_answers.append("A" if seq == 0 else "B")
            curr_images.append(images_dict[index][j])
        questions.append(curr_question)
        answers.append(curr_answers)
        images.append(curr_images)

    count = 0
    for i in range(len(images)):
        for j in range(len(images[i])):
            train.append(dict(
                id=f"counterfactuals_{count}",
                image=images[i][j], 
                conversations=[
                    {
                        "from": "human",
                        "value": questions[i][j]
                    },
                    {
                        "from": "gpt",
                        "value": answers[i][j]
                    }
                ]
            ))
            count += 1


for i, curr in enumerate(list_of_jsons):
    folder = folders[i]
    questions = []
    answers = []
    images = []
    for index, metas in tqdm(curr.items()):
        pos_img = os.path.join(input_folder, folder, "pos", f"{index}.jpg")
        for meta in metas:
            seq = random.randint(0, 1)
            image_id = meta["image_id"]
            neg_img = None
            for potential_file in [os.path.join(input_folder, folder, "neg", f"{index}_{image_id}.png"),
                                   os.path.join(input_folder, folder, "neg", f"{index}_{image_id}.jpg"),
                                   os.path.join(input_folder, folder, "neg", f"{index}.jpg")]:
                if os.path.isfile(potential_file):
                    neg_img = potential_file
                    break
            if neg_img is None: continue
            pos = meta["pos"]
            neg = meta["neg"]
            if seq == 1: pos, neg = neg, pos
            questions.append(f"<image>\nWhich of the following captions best describes the image?\nA. {pos}\nB. {neg}\nAnswer with the option's letter from the given choices directly.")
            answers.append(("A", "B") if seq == 0 else ("B", "A"))
            images.append((pos_img, neg_img))
    if want_test_split:
        images_train, images_test, questions_train, questions_test, answers_train, answers_test = train_test_split(images, questions, answers, test_size=0.2)
    else:
        images_train, questions_train, answers_train = images, questions, answers
    count = 0
    for i in range(len(images_train)):
        for j in range(len(images_train[i])):
            train.append(dict(
                id=f"{folder}_{count}",
                image=images_train[i][j], 
                conversations=[
                    {
                        "from": "human",
                        "value": questions_train[i]
                    },
                    {
                        "from": "gpt",
                        "value": answers_train[i][j]
                    }
                ]
            ))
            count += 1
    if want_test_split:
        for i in range(len(images_test)):
            for j in range(2):
                test.append(dict(
                    id=f"{folder}_{count}",
                    image=images_test[i][j],
                    conversations=[
                        {
                            "from": "human",
                            "value": questions_test[i]
                        },
                        {
                            "from": "gpt",
                            "value": answers_test[i][j]
                        }
                    ]
                ))
                count += 1

random.shuffle(train)
with open(os.path.join(output_folder, f"train_{output_name}.json"), 'w') as f:
    json.dump(train, f, indent=4)
if want_test_split:
    random.shuffle(test)
    with open(os.path.join(output_folder, f"test_{output_name}.json"), 'w') as f:
        json.dump(test, f, indent=4)