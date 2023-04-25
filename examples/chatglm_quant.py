# %%
import json
import torch
from chatglm_q.tokenizer import ChatGLMTokenizer

torch.manual_seed(42)

tokenizer = ChatGLMTokenizer("../models/chatglm-6b-safe/sentencepiece.model")
raw_data = json.load(open("../data/zh-data01.json"))

calibrate_data_size = 10
calibrate_data = torch.randperm(len(raw_data))[:calibrate_data_size].tolist()

def create_input(data):
    input_ids, prefix_mask = tokenizer.encode(
        f"[Round 0]\n指令：{data['instruction']}\n输入：{data['input']}\n回答：", 
        data['output'],
    )
    return torch.LongTensor([input_ids]), torch.LongTensor([prefix_mask])

data = [create_input(raw_data[idx]) for idx in calibrate_data]

# %%
from chatglm_q.loader import load_state_dict

model = load_state_dict("../models/chatglm-6b-safe")

# %%
from torch import nn
from chatglm_q.quantizer import get_quant_embedding, GPTQLinearQuantizer

model.word_embedding = get_quant_embedding(model.word_embedding)

# %%
from tqdm import tqdm

num_layers = model.config.num_layers

with torch.no_grad():
    current_h = [
        model.word_embedding(row[0])
        for row in tqdm(data)
    ]

    pe_and_mask = [
        model.prepare_pe_and_mask(emb, row[1], False)
        for emb, row in zip(current_h, data)
    ]

# %%

for layer_idx in tqdm(range(num_layers)):

    layer = model.layers[layer_idx]
    qlayers: dict[str, GPTQLinearQuantizer] = {}

    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            qlayers[name] = GPTQLinearQuantizer(module)

    next_h = tuple()
    for h, (pe, mask) in zip(current_h, pe_and_mask):
        with torch.no_grad():
            h, _ = layer(
                h,
                freqs_cis=pe,
                attention_mask=mask,
                kv_cache=None,
                use_past=False
            )
        next_h += (h,)

    current_h = next_h

    for name, module in qlayers.items():
        module.remove_hook()
        path = name.split(".")
        parent = layer.get_submodule(".".join(path[:-1]))
        setattr(parent, path[-1], module.get_quantized_linear(pring_loss=True))

del qlayers

# %%
lm_head_q = GPTQLinearQuantizer(model.lm_head)

with torch.no_grad():
    for h in current_h:
        model.lm_head(model.final_ln(h))

lm_head_q.remove_hook()
setattr(model, "lm_head", lm_head_q.get_quantized_linear(pring_loss=True))

del lm_head_q
del current_h

# %%
from safetensors.torch import save_file

save_file(model.state_dict(), "../models/chatglm-6b-int8.safetensors")

# %%
from chatglm_q.loader import load_quant_model

model = load_quant_model("../models/chatglm-6b-int8.safetensors")

# %%
