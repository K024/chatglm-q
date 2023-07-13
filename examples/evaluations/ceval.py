# %%
import json
from pathlib import Path

# CEval data from https://github.com/THUDM/ChatGLM2-6B/tree/main/evaluation
all_data = [
    (file.parent.name, file.stem, json.loads(line))
    for file in Path("../../data/CEval/val").rglob("*.jsonl")
    for line in file.read_text().splitlines()
    if len(line)
]

# %%
import torch
from chatglm_q.decoder import ChatGLMDecoder

activation = torch.float16
device = torch.device("cuda")
decoder = ChatGLMDecoder.from_pretrained("../../models/chatglm2-6b-safe", device, torch_dtype=activation)

# %%
choice_tokens = [decoder.tokenizer[choice] for choice in "ABCD"]
think_template = "[Round 1]\n\n问：{}\n\n答："
final_template = "[Round 1]\n\n问：{}\n\n答：{}\n综上所述，正确的选项是："
direct_template = "[Round 1]\n\n问：{}\n\n答：正确的选项是："
chain_of_thoughts = False

# %%
from tqdm.auto import tqdm

total = 0
corrects = 0
evaluations = []
progress_bar = tqdm(all_data)

for category, test_name, data in progress_bar:
    question = data["inputs_pretokenized"]

    if chain_of_thoughts:
        thoughts = list(decoder.generate(
            think_template.format(question),
            temperature=0.5
        ))[-1]
        prompt = final_template.format(question, thoughts),
    else:
        prompt = direct_template.format(question)

    with torch.no_grad():
      _, model_output, _ = decoder.model(
          **decoder.tokenizer(
              prompt,
              padding=True,
              return_tensors="pt",
          ).to(device),
      )
      model_choices = model_output[0, -1, choice_tokens]
      model_predict = torch.argmax(model_choices).item()
      correct = int(model_predict == data['label'])

    evaluations.append((category, correct))

    total += 1
    corrects += correct
    progress_bar.set_postfix_str(f"{category} {corrects}/{total} {corrects/total:.2%}")

# %%
print(f"{'total': <16}: {corrects}/{total} {corrects/total:.2%}")
print(f"-------")

categories = { cat: [] for cat in sorted(set(data[0] for data in evaluations)) }
for cat, correct in evaluations:
    categories[cat].append(int(correct))

for cat_name, cat_list in categories.items():
    t = len(cat_list)
    c = sum(cat_list)
    print(f"{cat_name: <16}: {c}/{t} {c/t:.2%}")

# %%
