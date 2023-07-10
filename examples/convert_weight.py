# %%
from huggingface_hub import snapshot_download

target_path = "../models/chatglm2-6b-safe"
path_or_repo_id = "https://huggingface.co/THUDM/chatglm2-6b"
cache_dir = None
token = None

model_path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)

# %%
from pathlib import Path

model_path = Path(model_path)
target_path = Path(target_path)
target_path.mkdir(parents=True, exist_ok=True)

# %%
name_mapping = {
    'transformer.embedding.word_embeddings.weight': 'word_embedding.weight',
    'transformer.encoder.final_layernorm.weight': 'final_ln.weight',
    'transformer.output_layer.weight': 'lm_head.weight'
}

for i in range(28):
    name_mapping.update({
        f'transformer.encoder.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
        f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight': f'layers.{i}.attn.qkv_proj.weight',
        f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias': f'layers.{i}.attn.qkv_proj.bias',
        f'transformer.encoder.layers.{i}.self_attention.dense.weight': f'layers.{i}.attn.o_proj.weight',
        f'transformer.encoder.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
        f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight': f'layers.{i}.ffn.w_in.weight',
        f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight': f'layers.{i}.ffn.w_out.weight',
    })

# %%
import json
import shutil
import torch
from tqdm.auto import tqdm
from collections import OrderedDict
from safetensors.torch import save_file
from chatglm_q.loader import ChatGLMLoadConfig

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

config = ChatGLMLoadConfig(
    weight_files = [bin_file.replace(".bin", ".safetensors") for bin_file in bin_files],
    torch_dtype="float16",
)

shutil.copy(model_path / "tokenizer.model", target_path / config.tokenizer_file)

config_path = target_path / "config.json"
config_path.write_text(config.to_json())

# %%
