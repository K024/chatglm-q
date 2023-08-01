import json
import shutil
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Literal, Union, Any
from pathlib import Path

import torch
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, safe_open

from . import chatglm as chatglm_model
from . import internlm as internlm_model
modules = [chatglm_model, internlm_model]
model_types = Union[chatglm_model.ChatGLM2Model, internlm_model.InternLMModel]
tokenizer_types = Union[chatglm_model.ChatGLM2Tokenizer, internlm_model.InternLMTokenizer]


@dataclass
class LoadConfig():
    model_type: Literal["ChatGLM2Model", "InternLMModel"]
    model_config: Union[chatglm_model.ChatGLM2Config, internlm_model.InternLMConfig]
    quant_type: Literal["none", "int8", "int4g32"] = "none"
    weight_files: list[str] = field(default_factory=list)
    tokenizer_file: str = "sentencepiece.model"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    def get_model_classes(self):
        model_classes = [x for x in modules if x.model_class.__name__ == self.model_type]
        assert len(model_classes) == 1, f"Unsupported model_type '{self.model_type}'"
        return model_classes[0]

    def __post_init__(self):
        model_classes = self.get_model_classes()
        if not isinstance(self.model_config, model_classes.config_class):
            self.model_config = model_classes.config_class(**self.model_config)

    def get_torch_dtype(self):
        return getattr(torch, self.torch_dtype)

    @staticmethod
    def from_json(json_str):
        return LoadConfig(**json.loads(json_str))

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@torch.no_grad()
def load_state_dict(state_dict: dict[str, torch.Tensor], files: list[Path]):
    for file in files:
        with safe_open(file, framework="pt") as f:
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


def load_model_and_tokenizer(
    model_path: Union[str, Path], torch_dtype=None, load_model=True, load_tokenizer=True,
) -> tuple[LoadConfig, model_types, tokenizer_types]:

    model_path = Path(model_path)
    config_path = model_path / "config.json"
    config = LoadConfig.from_json(config_path.read_bytes())
    torch_dtype = torch_dtype or config.get_torch_dtype()

    model_classes = config.get_model_classes()

    model = None
    if load_model:
        if config.quant_type == "none":
            model = model_classes.model_class(config.model_config, torch_dtype)
        elif config.quant_type == "int8":
            model = model_classes.create_int8_model(config.model_config, torch_dtype)
        elif config.quant_type == "int4g32":
            model = model_classes.create_int4_model(config.model_config, 32, torch_dtype)
        else:
            raise NotImplementedError(f"No quant_type named '{config.quant_type}'")

        state_dict = dict(**model.state_dict())
        files = config.weight_files if len(config.weight_files) == 1 else tqdm(config.weight_files)
        load_state_dict(state_dict, [model_path / file for file in files])

    tokenizer = None
    if load_tokenizer:
        tokenizer = model_classes.tokenizer_class(model_path / config.tokenizer_file)

    return config, model, tokenizer


def save_model_and_tokenizer(
    path: Union[str, Path],
    config: LoadConfig,
    model: Any,
    tokenizer: Any,
    shard=True,
    max_shard_bytes=2 * 1024 ** 3
):
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

        config.weight_files = sorted(set(weight_mapping.values()))

        for file in tqdm(config.weight_files):
            weights = { name: state_dict[name] for name, f in weight_mapping.items() if file == f }
            save_file(weights, path / file)

    config_path = path / "config.json"
    config_path.write_text(config.to_json())


def from_pretrained(path_or_repo_id: Union[Path, str], device=None, torch_dtype=None, cache_dir=None, token=None):
    path = Path(path_or_repo_id)
    if not path.exists() or not path.is_dir():
        assert isinstance(path_or_repo_id, str)
        path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)
    config, model, tokenizer = load_model_and_tokenizer(path, torch_dtype)
    model_classes = config.get_model_classes()
    model.to(device=device)
    return model_classes.decoder_class(config, model, tokenizer, device=device)


def save_pretrained(decoder, path: Union[Path, str], shard=True):
    save_model_and_tokenizer(path, decoder.config, decoder.model, decoder.tokenizer, shard=shard)
