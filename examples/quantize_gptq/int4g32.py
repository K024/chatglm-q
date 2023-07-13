# %%
import json
import torch
from pathlib import Path
from chatglm_q.loader import ChatGLMLoadConfig, load_model_and_tokenizer, save_model_and_tokenizer

torch.manual_seed(42)

_, model, tokenizer = load_model_and_tokenizer("../../models/chatglm2-6b-safe", torch.float32)

# CEval data from https://github.com/THUDM/ChatGLM2-6B/tree/main/evaluation
all_data = [
    json.loads(line)
    for file in Path("../../data/CEval/val").rglob("*.jsonl")
    for line in file.read_text().splitlines()
    if len(line)
]

batch_size = 20
calibrate_data_size = 200
calibrate_data_idx = torch.randperm(len(all_data))[:calibrate_data_size] \
    .view(-1, batch_size).tolist()

data = [
    tokenizer([
        f"问：{all_data[idx]['inputs_pretokenized']}\n\n"
        f"答：{all_data[idx]['targets_pretokenized'][0]}"
        for idx in batch
    ], padding=True, return_tensors="pt")
    for batch in calibrate_data_idx
]

# %%
from torch import nn
from chatglm_q.int4.quantizer import get_quant_embedding, GPTQLinearQuantizer

device = torch.device("cuda")
# later move to device layer by layer
# model.to(device)

model.word_embedding = get_quant_embedding(model.word_embedding)

# %%
num_layers = model.config.num_layers

with torch.no_grad():
    prepared_input = [
        model.prepare_input(**batch)
        for batch in data
    ]
    current_h = [batch[0] for batch in prepared_input]

# %%
from tqdm.auto import tqdm

for layer_idx in tqdm(range(num_layers)):

    layer = model.layers[layer_idx]
    layer.to(device)
    qlayers: dict[str, GPTQLinearQuantizer] = {}

    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            qlayers[name] = GPTQLinearQuantizer(module)

    next_h = tuple()
    for h, (_, mask, pe) in zip(current_h, prepared_input):
        with torch.no_grad():
            h, _ = layer(
                x=h.to(device),
                attention_mask=mask.to(device),
                freqs_cis=pe.to(device),
                kv_cache=None,
            )
        next_h += (h,)

    current_h = next_h

    for name, module in qlayers.items():
        module.remove_hook()
        parent_path, module_name = name.rsplit(".", 1)
        parent = layer.get_submodule(parent_path)
        setattr(parent, module_name, module.get_quantized_linear(pring_loss=True))

    model.to("cpu")
    layer.to("cpu")

del qlayers

# %%
model.final_ln.to(device)
model.lm_head.to(device)
lm_head_q = GPTQLinearQuantizer(model.lm_head)

with torch.no_grad():
    for h in current_h:
        model.lm_head(model.final_ln(h))

lm_head_q.remove_hook()
setattr(model, "lm_head", lm_head_q.get_quantized_linear(pring_loss=True))

model.to("cpu")
del lm_head_q
del current_h

# %%
# set torch_dtype (activation type) as needed
config = ChatGLMLoadConfig(model_config=model.config, quant_type="int4g32", torch_dtype="float16")

save_model_and_tokenizer("../../models/chatglm2-6b-int4g32", config, model, tokenizer)
# %%
