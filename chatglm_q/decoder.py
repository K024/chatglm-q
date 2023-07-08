import re
import time
import torch
from pathlib import Path
from typing import Union
from huggingface_hub import snapshot_download
from .model import ChatGLM2Model
from .tokenizer import ChatGLM2Tokenizer
from .loader import ChatGLMLoadConfig, load_model_and_tokenizer, save_model_and_tokenizer


def top_p_sampling(logits: torch.Tensor, top_k=100, top_p=0.8, temperature=1.0):
    # top_k
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    probs, indices = torch.sort(probs, dim=-1, descending=True)
    probs = probs[..., :top_k]
    indices = indices[..., :top_k]

    # top_p
    cumsum = torch.cumsum(probs, dim=-1)
    probs[(cumsum - probs) > top_p] = 0.0
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # sample
    next_token = torch.multinomial(probs, num_samples=1)
    output = torch.gather(indices, dim=-1, index=next_token)
    return output[..., 0]


class ChatGLMDecoder():
    def __init__(
        self,
        config: ChatGLMLoadConfig,
        model: ChatGLM2Model,
        tokenizer: ChatGLM2Tokenizer,
        eos_token = "</s>",
        device = None,
        max_sequence_length: int = None,
        time_log = False,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = tokenizer[eos_token]
        self.max_sequence_length = max_sequence_length or config.model_config.max_sequence_length
        self.time_log = time_log


    @staticmethod
    def from_pretrained(path_or_repo_id: Union[Path, str], device=None, torch_dtype=None, cache_dir=None, token=None):
        path = Path(path_or_repo_id)
        if not path.exists() or not path.is_dir():
            assert isinstance(path_or_repo_id, str)
            path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)
        config, model, tokenizer = load_model_and_tokenizer(path, torch_dtype)
        model.to(device=device)
        return ChatGLMDecoder(config, model, tokenizer, device=device)


    def save_pretrained(self, path: Union[Path, str], shard=True):
        save_model_and_tokenizer(path, self.config, self.model, self.tokenizer, shard=shard)


    def generate(self, prefix_text: str, max_generated_tokens=400, top_k=100, top_p=0.8, temperature=1.0):
        model, tokenizer = self.model, self.tokenizer
        eos_token_id = self.eos_token_id

        prefix_ids = tokenizer.encode(prefix_text)
        input_ids = torch.LongTensor([prefix_ids])
        past_key_values = None

        generated_tokens = []
        generate_time = []

        while len(generated_tokens) < max_generated_tokens \
            and input_ids.shape[1] < self.max_sequence_length:

            with torch.no_grad():
                start_time = time.perf_counter()
                _, logits, past_key_values = model(
                    input_ids=input_ids.to(self.device),
                    past_key_values=past_key_values,
                    incremental_generate=True,
                )
                next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()
                end_time = time.perf_counter()
                generate_time.append(end_time - start_time)

            generated_tokens += [next_token]
            if next_token == eos_token_id:
                break

            response_text = process_response(tokenizer.decode(generated_tokens))
            if response_text and response_text[-1] != "�":
                yield response_text

            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]]).long(),
            ], dim=1)

        if self.time_log:
            init_time, *rest_time = generate_time
            print(f"Decoder:")
            print(f"  len: {len(prefix_ids)}(prefix) + {len(generated_tokens)}(gen)")
            print(f" init: {init_time:.6f} s")
            print(f"  sum: {sum(generate_time):.6f} s")
            print(f"  gen: {len(rest_time) / sum(rest_time):.6f} tok/s")
            print(f"  avg: {len(generate_time) / sum(generate_time):.6f} tok/s")

        return process_response(tokenizer.decode(generated_tokens))


def chat_template(history: list[tuple[str, str]], current: str):
    prompt = ""
    chat_round = 0
    for question, answer in history:
        prompt += f"[Round {chat_round}]\n\n问：{question}\n\n答：{answer}\n\n"
        chat_round += 1
    prompt += f"[Round {chat_round}]\n\n问：{current}\n\n答："
    return prompt


def process_response(response: str):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response
