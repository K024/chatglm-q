# %%
import torch
from torch import nn
from tqdm.auto import tqdm
from chatglm_q.loader import load_state_dict
from chatglm_q.tokenizer import ChatGLMTokenizer
from chatglm_q.decoder import ChatGLMDecoder

model = load_state_dict("../models/chatglm-6b-safe")
tokenizer = ChatGLMTokenizer("../models/chatglm-6b-safe/sentencepiece.model")
decoder = ChatGLMDecoder(model, tokenizer)

# %%
from chatglm_q.quantizer import get_quant_int8_linear, get_quant_embedding

model.word_embedding = get_quant_embedding(model.word_embedding)

# %%
linear_layers: dict[str, nn.Linear] = {}

for name, module in model.named_modules():
    if isinstance(module, nn.Linear): # and "lm_head" not in name:
        linear_layers[name] = module

for name, module in tqdm(linear_layers.items()):
    module_path = name.split(".")
    parent = model.get_submodule(".".join(module_path[:-1]))

    module = get_quant_int8_linear(module)
    setattr(parent, module_path[-1], module)

# %%
from safetensors.torch import save_file

save_file(model.state_dict(), "../models/chatglm-6b-int8-naive.safetensors")

# %%
