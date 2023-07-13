# %%
import json
from pathlib import Path

# CEval data from https://github.com/THUDM/ChatGLM2-6B/tree/main/evaluation
all_data = [
    json.loads(line)["inputs_pretokenized"]
    for file in Path("../../data/CEval/val").rglob("*.jsonl")
    for line in file.read_text().splitlines()
    if len(line)
]

batch_size = 20
all_data = [
    all_data[idx:idx + batch_size]
    for idx in range(0, len(all_data), batch_size)
]

# %%
import torch
from chatglm_q.loader import load_model_and_tokenizer

device = torch.device("cuda")
torch_dtype = torch.float16
_, model, tokenizer = load_model_and_tokenizer("../../models/chatglm2-6b-safe", torch_dtype)
model = model.to(device)

# %%
from tqdm.auto import tqdm

losses = []
progress_bar = tqdm(all_data)

for texts in progress_bar:
    inputs = tokenizer(texts, padding=True, return_tensors="pt", return_labels=True)

    with torch.no_grad():
        loss, _, _ = model(**inputs.to(device))
        losses.append(loss.item())

# %%
import math

avg = sum(losses) / len(losses)
print(f"ppl: {math.exp(avg):.6f}")

# %%
