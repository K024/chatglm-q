# %%
import json
import torch
from chatglm_q.loader import ChatGLMLoadConfig, load_model_and_tokenizer, save_model_and_tokenizer

torch.manual_seed(42)

_, model, tokenizer = load_model_and_tokenizer("../models/chatglm-6b-safe", torch.float32)

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
from torch import nn
from chatglm_q.int8.quantizer import get_quant_embedding, GPTQLinearQuantizer

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
        parent_path, module_name = name.rsplit(".", 1)
        parent = layer.get_submodule(parent_path)
        setattr(parent, module_name, module.get_quantized_linear(pring_loss=True))

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
config = ChatGLMLoadConfig(quant_type="int8")

save_model_and_tokenizer("../models/chatglm-6b-int8", config, model, tokenizer)

# %%
