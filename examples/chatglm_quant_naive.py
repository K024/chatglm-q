# %%
import torch
from torch import nn
from tqdm.auto import tqdm
from chatglm_q.loader import ChatGLMLoadConfig, load_model_and_tokenizer, save_model_and_tokenizer

_, model, tokenizer = load_model_and_tokenizer("../models/chatglm-6b-safe", torch.float32)

# %%
from chatglm_q.int8.quantizer import get_quant_int8_linear, get_quant_embedding

model.word_embedding = get_quant_embedding(model.word_embedding)

# %%
linear_layers: dict[str, nn.Linear] = {}

for name, module in model.named_modules():
    if isinstance(module, nn.Linear): # and "lm_head" not in name:
        linear_layers[name] = module

for name, module in tqdm(linear_layers.items()):
    if "." in name:
        parent_path, module_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
    else:
        module_name = name
        parent = model

    module = get_quant_int8_linear(module)
    setattr(parent, module_name, module)

# %%
config = ChatGLMLoadConfig(quant_type="int8")

save_model_and_tokenizer("../models/chatglm-6b-int8-naive", config, model, tokenizer)

# %%
