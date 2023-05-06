import json
import shutil
from pathlib import Path
from collections import OrderedDict
import torch
from tqdm.auto import tqdm
from safetensors.torch import save_file, load_file, safe_open
from .model import ChatGLMModel, ChatGLMConfig


def create_quant_int8_model():
    try:
        from . import model as modeling
        from .int8.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLMModel(ChatGLMConfig())
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


def create_quant_int4_model():
    try:
        from . import model as modeling
        from .int4.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLMModel(ChatGLMConfig())
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


@torch.no_grad()
def load_quant_model(state_dict_path: str, quant_type = "int8"):
    if quant_type == "int8":
        model = create_quant_int8_model()
    elif quant_type == "int4":
        model = create_quant_int4_model()
    else:
        raise NotImplementedError(f"No quant_type named '{quant_type}'")

    state_dict = dict(**model.state_dict())

    with safe_open(state_dict_path, framework="pt") as f:
        for k in f.keys():
            if k not in state_dict:
                print(f'"{k}" is ignored')
                continue
            v = f.get_tensor(k)
            if state_dict[k].is_floating_point():
                v = v.type_as(state_dict[k])
            state_dict[k].copy_(v.to(state_dict[k].device))
            state_dict.pop(k)

    if len(state_dict):
        print(f'model weights "{", ".join(state_dict.keys())}" are not initialized')

    return model


@torch.no_grad()
def load_state_dict(model_path: str, model: ChatGLMModel = None, dtype = None):
    if model is None:
        model = ChatGLMModel(ChatGLMConfig(), dtype)
    state_dict = dict(**model.state_dict())

    model_path = Path(model_path)

    for i in tqdm(range(8), "Loading"):
        part = load_file(model_path / f'pytorch_model-0000{i + 1}-of-00008.safetensors')

        for k, v in part.items():
            if k not in state_dict:
                print(f'"{k}" is ignored')
                continue
            if state_dict[k].is_floating_point():
                v = v.type_as(state_dict[k])
            state_dict[k].copy_(v.to(state_dict[k].device))
            state_dict.pop(k)

    if len(state_dict):
        print(f'model weights "{", ".join(state_dict.keys())}" are not initialized')

    return model


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
                continue
            new_state_dict[name_mapping[k]] = v

        save_file(new_state_dict, target_path / bin_file.replace(".bin", ".safetensors"))

    shutil.copy(model_path / "ice_text.model", target_path / "sentencepiece.model")
