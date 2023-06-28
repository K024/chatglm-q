import json
import shutil
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Literal, Union
from pathlib import Path

import torch
from tqdm.auto import tqdm
from safetensors.torch import save_file, safe_open
from .model import ChatGLMModel, ChatGLMConfig
from .tokenizer import ChatGLMTokenizer


@dataclass
class ChatGLMLoadConfig():
    model_type: Literal["ChatGLMModel"] = "ChatGLMModel"
    model_config: ChatGLMConfig = field(default_factory=ChatGLMConfig)
    quant_type: Literal["none", "int8", "int4g32"] = "none"
    weight_files: list[str] = field(default_factory=list)
    tokenizer_file: str = "sentencepiece.model"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    def __post_init__(self):
        assert self.model_type == "ChatGLMModel", "Only 'ChatGLMModel' is supported"
        if not isinstance(self.model_config, ChatGLMConfig):
            self.model_config = ChatGLMConfig(**self.model_config)

    def get_torch_dtype(self):
        return getattr(torch, self.torch_dtype)

    @staticmethod
    def from_json(json_str):
        return ChatGLMLoadConfig(**json.loads(json_str))

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def create_quant_int8_model(config = ChatGLMConfig(), dtype=None):
    try:
        from . import model as modeling
        from .int8.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLMModel(config, dtype)
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


def create_quant_int4_model(config=ChatGLMConfig(), group_size=32, dtype=None):
    try:
        from . import model as modeling
        from .int4 import qlinear
        from .int4.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_group_size = qlinear.DEFAULT_GROUP_SIZE
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        qlinear.DEFAULT_GROUP_SIZE = group_size
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLMModel(config, dtype)
    finally:
        qlinear.DEFAULT_GROUP_SIZE = prev_group_size
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


@torch.no_grad()
def load_model_and_tokenizer(model_path: Union[str, Path], torch_dtype=None):
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    config = ChatGLMLoadConfig.from_json(config_path.read_bytes())
    torch_dtype = torch_dtype or config.get_torch_dtype()

    if config.quant_type == "none":
        model = ChatGLMModel(config.model_config, torch_dtype)
    elif config.quant_type == "int8":
        model = create_quant_int8_model(config.model_config, torch_dtype)
    elif config.quant_type == "int4g32":
        model = create_quant_int4_model(config.model_config, 32, torch_dtype)
    else:
        raise NotImplementedError(f"No quant_type named '{config.quant_type}'")

    state_dict = dict(**model.state_dict())
    files = config.weight_files if len(config.weight_files) == 1 else tqdm(config.weight_files)

    for file in files:
        with safe_open(model_path / file, framework="pt") as f:
            for k in f.keys():
                try:
                    if k not in state_dict:
                        print(f'"{k}" is ignored')
                        continue
                    v = f.get_tensor(k)
                    if state_dict[k].is_floating_point():
                        v = v.type_as(state_dict[k])
                    state_dict[k].copy_(v.to(state_dict[k].device))
                    state_dict.pop(k)
                except:
                    print(f"error handling weight '{k}'")
                    raise

    if len(state_dict):
        print(f'model weights "{", ".join(state_dict.keys())}" are not initialized')
    
    tokenizer = ChatGLMTokenizer(model_path / config.tokenizer_file)

    return config, model, tokenizer


def save_model_and_tokenizer(path: Union[str, Path], config: ChatGLMLoadConfig, model: ChatGLMModel, tokenizer: ChatGLMTokenizer, shard=True, max_shard_bytes=2 * 1024 ** 3):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    else:
        assert path.is_dir()
    tokenizer_path = path / config.tokenizer_file
    shutil.copy(tokenizer.vocab_file, tokenizer_path)

    if not shard:
        config.weight_files = ["model_weights.safetensors"]
        save_file(model.state_dict(), path / config.weight_files[0])

    else:
        weight_mapping = {}
        current_index = 0
        current_size = 0
        state_dict = model.state_dict()
        for name, weight in state_dict.items():
            size = weight.element_size() * weight.numel()
            if current_size + size > max_shard_bytes:
                current_index += 1
                current_size = 0
            current_size += size
            weight_mapping[name] = f"model_weights_{current_index}.safetensors"

        config.weight_files = list(set(weight_mapping.values()))

        for file in tqdm(config.weight_files):
            weights = { name: state_dict[name] for name, f in weight_mapping.items() if file == f }
            save_file(weights, path / file)

    config_path = path / "config.json"
    config_path.write_text(config.to_json())


@torch.no_grad()
def convert_transformers_weights(model_path, target_path):
    name_mapping = {
        'transformer.word_embeddings.weight': 'word_embedding.weight',
        'transformer.final_layernorm.weight': 'final_ln.weight',
        'transformer.final_layernorm.bias': 'final_ln.bias',
        'lm_head.weight': 'lm_head.weight'
    }

    for i in range(28):
        name_mapping.update({
            f'transformer.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
            f'transformer.layers.{i}.input_layernorm.bias': f'layers.{i}.attn_ln.bias',
            f'transformer.layers.{i}.attention.query_key_value.weight': f'layers.{i}.attn.qkv_proj.weight',
            f'transformer.layers.{i}.attention.query_key_value.bias': f'layers.{i}.attn.qkv_proj.bias',
            f'transformer.layers.{i}.attention.dense.weight': f'layers.{i}.attn.o_proj.weight',
            f'transformer.layers.{i}.attention.dense.bias': f'layers.{i}.attn.o_proj.bias',
            f'transformer.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
            f'transformer.layers.{i}.post_attention_layernorm.bias': f'layers.{i}.ffn_ln.bias',
            f'transformer.layers.{i}.mlp.dense_h_to_4h.weight': f'layers.{i}.ffn.w_in.weight',
            f'transformer.layers.{i}.mlp.dense_h_to_4h.bias': f'layers.{i}.ffn.w_in.bias',
            f'transformer.layers.{i}.mlp.dense_4h_to_h.weight': f'layers.{i}.ffn.w_out.weight',
            f'transformer.layers.{i}.mlp.dense_4h_to_h.bias': f'layers.{i}.ffn.w_out.bias',
        })

    model_path = Path(model_path)
    target_path = Path(target_path)

    indices = json.load(open(model_path / "pytorch_model.bin.index.json"))
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
        weight_files = [bin_file.replace(".bin", ".safetensors") for bin_file in bin_files]
    )

    shutil.copy(model_path / "ice_text.model", target_path / config.tokenizer_file)

    config_path = target_path / "config.json"
    config_path.write_text(config.to_json())
