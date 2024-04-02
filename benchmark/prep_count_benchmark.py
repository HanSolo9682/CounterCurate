import json
import os

pointqa_path = "../datasets/pointingqa"
output_folder = "."
with open(os.path.join(pointqa_path, "Datasets", "LookTwiceQA", "looktwiceqa_dataset.json"), 'r') as f:
    ds = json.load(f)

vg_dict = {}
indexes_to_remove = set()
for index, items in ds.items():
    for item in items:
        id, name, question, ans = item['id'], item['obj_type'], item['obj_question'], item['ans']
        if id not in vg_dict.keys():
            vg_dict[id] = []
        for question in vg_dict[id]:
            if question["name"] == name:
                indexes_to_remove.add(id)
                break
        vg_dict[id].append(dict(name=name, ans=ans))

for index in indexes_to_remove:
    del vg_dict[index]


with open(os.path.join(output_folder, "vg_dict.json"), 'w') as f:
    json.dump(vg_dict, f, indent=4)