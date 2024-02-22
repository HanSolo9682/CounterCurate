import json
import os

input_folder = "./Flickr30k-Counterfactuals"
output_folder = "."


with open(os.path.join(input_folder, "gpt4v_neg_prompt.json"), 'r') as f:
    items = json.load(f)

results = {}
for item in items:
    idx, query, conv = item["idx"], item["query"], item["conv"]
    idx, subidx = idx.split('-')
    if subidx != '0': continue
    try:
        content: str = conv["choices"][0]["message"]["content"]
        if "I'm sorry" in content or len(content) == 0: continue
        content = content.replace("(", "[").replace(")", "]").replace("None", "null").replace("Your answer:\n", '')
        json_content = json.loads(content)
    except Exception as e:
        continue

    try:
        results[idx] = dict(
            noun=json_content["noun"]["caption"],
            adjective=json_content["adjective"]["caption"],
            reverse=json_content["reverse"]["caption"] if json_content["reverse"] not in ["None", "null", None] else None,
        )
    except Exception as e:
        break
    
with open(os.path.join(output_folder, "dalle_gen_counterfactuals.json"), 'w') as f:
    json.dump(results, f, indent=4)