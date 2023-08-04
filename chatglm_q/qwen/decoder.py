import time
import torch
from .model import QwenModel
from .tokenizer import QwenTokenizer


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


class QwenDecoder():
    def __init__(
        self,
        config,
        model: QwenModel,
        tokenizer: QwenTokenizer,
        eos_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        device = None,
        max_sequence_length: int = None,
        time_log = False,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if not isinstance(eos_tokens, list):
            eos_tokens = [eos_tokens]
        self.eos_token_ids = [tokenizer[t] for t in eos_tokens]
        self.max_sequence_length = max_sequence_length or model.config.max_sequence_length
        self.time_log = time_log

    def generate(self, prefix_text: str, max_generated_tokens=400, top_k=100, top_p=0.8, temperature=1.0):
        model, tokenizer = self.model, self.tokenizer
        eos_token_ids = self.eos_token_ids

        prefix_ids = tokenizer.encode(prefix_text)
        input_ids = torch.LongTensor([prefix_ids])
        past_key_values = None

        generated_tokens = []
        generate_time = []

        while len(generated_tokens) < max_generated_tokens \
            and len(generated_tokens) + len(prefix_ids) < self.max_sequence_length:

            with torch.no_grad():
                start_time = time.perf_counter()
                _, logits, past_key_values = model(
                    input_ids=input_ids.to(self.device),
                    past_key_values=past_key_values,
                )
                next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()
                end_time = time.perf_counter()
                generate_time.append(end_time - start_time)

            generated_tokens += [next_token]
            if next_token in eos_token_ids:
                break

            response_text = tokenizer.decode(generated_tokens)
            if response_text and response_text[-1] != "�":
                yield response_text

            input_ids = torch.tensor([[next_token]]).long()

        if self.time_log:
            init_time, *rest_time = generate_time
            print(f"Decoder:")
            print(f"  len: {len(prefix_ids)}(prefix) + {len(generated_tokens)}(gen)")
            print(f" init: {init_time:.6f} s")
            print(f"  sum: {sum(generate_time):.6f} s")
            print(f"  gen: {len(rest_time) / sum(rest_time):.6f} tok/s")
            print(f"  avg: {len(generate_time) / sum(generate_time):.6f} tok/s")

        return tokenizer.decode(generated_tokens)


    def chat(self, history: list[tuple[str, str]], question: str, **kwargs):
        return self.generate(chat_template(history, question), **kwargs)


def chat_template(history: list[tuple[str, str]], query: str):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    for q, a in history:
        prompt += f"\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    prompt += f"\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
