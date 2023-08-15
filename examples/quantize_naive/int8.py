# %%
import torch
from torch import nn
from tqdm.auto import tqdm
from chatglm_q.loader import LoadConfig, load_model_and_tokenizer, save_model_and_tokenizer

model_path = "../../models/chatglm2-6b-safe"
config, model, tokenizer = load_model_and_tokenizer(model_path, torch.float32)

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
# set torch_dtype (activation type) as needed
config = LoadConfig(
    model_type=config.model_type,
    model_config=model.config,
    quant_type="int8",
    torch_dtype="float16",
    tokenizer_file=config.tokenizer_file,
)

save_model_and_tokenizer(model_path.rstrip("/\\") + "-int8-naive", config, model, tokenizer)

# %%
