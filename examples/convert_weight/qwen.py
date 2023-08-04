# %%
from huggingface_hub import snapshot_download

target_path = "../../models/qwen-7b-chat-safe"
path_or_repo_id = "Qwen/Qwen-7B-Chat"
cache_dir = None
token = None

# model_path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)
model_path = "/home/onedev/nvdocker-sshd/chatglm-q/models/Qwen-7B-Chat"

# %%
from pathlib import Path

model_path = Path(model_path)
target_path = Path(target_path)
target_path.mkdir(parents=True)

# %%
name_mapping = {
    'transformer.wte.weight': 'word_embedding.weight',
    'transformer.ln_f.weight': 'final_ln.weight',
    'lm_head.weight': 'lm_head.weight'
}

for i in range(32):
    name_mapping.update({
        f'transformer.h.{i}.ln_1.weight': f'layers.{i}.attn_ln.weight',
        f'transformer.h.{i}.attn.c_attn.weight': f'layers.{i}.attn.qkv_proj.weight',
        f'transformer.h.{i}.attn.c_attn.bias': f'layers.{i}.attn.qkv_proj.bias',
        f'transformer.h.{i}.attn.c_proj.weight': f'layers.{i}.attn.o_proj.weight',
        f'transformer.h.{i}.ln_2.weight': f'layers.{i}.ffn_ln.weight',
        f'transformer.h.{i}.mlp.w1.weight': f'layers.{i}.ffn.w_in.weight',
        f'transformer.h.{i}.mlp.w2.weight': f'layers.{i}.ffn.w_gate.weight',
        f'transformer.h.{i}.mlp.c_proj.weight': f'layers.{i}.ffn.w_out.weight',
    })

# %%
import json
import shutil
import torch
from tqdm.auto import tqdm
from collections import OrderedDict
from safetensors.torch import save_file
from chatglm_q.loader import LoadConfig

bin_files = set(["pytorch_model.bin"])

for bin_file in tqdm(bin_files):
    state_dict = torch.load(model_path / bin_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k not in name_mapping:
            print(f"Unused weight '{k}'")
            continue
        new_state_dict[name_mapping[k]] = v

    save_file(new_state_dict, target_path / bin_file.replace(".bin", ".safetensors"))

config = LoadConfig(
    model_type="QwenModel",
    model_config={},
    weight_files = [bin_file.replace(".bin", ".safetensors") for bin_file in bin_files],
    torch_dtype="bfloat16",
    tokenizer_file="qwen.tiktoken",
)

shutil.copy(model_path / "qwen.tiktoken", target_path / config.tokenizer_file)

config_path = target_path / "config.json"
config_path.write_text(config.to_json())

# %%
