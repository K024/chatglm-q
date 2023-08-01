# %%
from huggingface_hub import snapshot_download

target_path = "../../models/internlm-chat-7b-safe"
path_or_repo_id = "internlm/internlm-chat-7b"
cache_dir = None
token = None

model_path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)

# %%
from pathlib import Path

model_path = Path(model_path)
target_path = Path(target_path)
target_path.mkdir(parents=True)

# %%
name_mapping = {
    'model.embed_tokens.weight': 'word_embedding.weight',
    'model.norm.weight': 'final_ln.weight',
    'lm_head.weight': 'lm_head.weight'
}

for i in range(32):
    name_mapping.update({
        f'model.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
        f'model.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
        f'model.layers.{i}.mlp.up_proj.weight': f'layers.{i}.ffn.w_in.weight',
        f'model.layers.{i}.mlp.gate_proj.weight': f'layers.{i}.ffn.w_gate.weight',
        f'model.layers.{i}.mlp.down_proj.weight': f'layers.{i}.ffn.w_out.weight',
    })
    for name in "qkvo":
        name_mapping.update({            
            f'model.layers.{i}.self_attn.{name}_proj.weight': f'layers.{i}.attn.{name}_proj.weight',
            f'model.layers.{i}.self_attn.{name}_proj.bias': f'layers.{i}.attn.{name}_proj.bias',
        })

# %%
import json
import shutil
import torch
from tqdm.auto import tqdm
from collections import OrderedDict
from safetensors.torch import save_file
from chatglm_q.loader import LoadConfig

indices = json.loads((model_path / "pytorch_model.bin.index.json").read_bytes())
bin_files = set(indices["weight_map"].values())

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
    model_type="InternLMModel",
    model_config={},
    weight_files = [bin_file.replace(".bin", ".safetensors") for bin_file in bin_files],
    torch_dtype="bfloat16",
)

shutil.copy(model_path / "tokenizer.model", target_path / config.tokenizer_file)

config_path = target_path / "config.json"
config_path.write_text(config.to_json())

# %%
