import re
import torch
from .model import ChatGLMModel
from .tokenizer import ChatGLMTokenizer


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
        model: ChatGLMModel,
        tokenizer: ChatGLMTokenizer,
        eos_token = "<eop>",
        top_k = 50,
        top_p = 0.8,
        temperature = 1.0,
        device = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.device = device
        self.eos_token_id = tokenizer[eos_token]

    @torch.no_grad()
    def generate(self, prefix_text: str, max_generated_tokens = 200, top_k = None, top_p = None, temperature = None):
        model, tokenizer = self.model, self.tokenizer
        top_k, top_p, temperature = top_k or self.top_k, top_p or self.top_p, temperature or self.temperature
        eos_token_id = self.eos_token_id

        input_ids, input_prefix_mask = tokenizer.encode(prefix_text)
        input_ids = torch.LongTensor([input_ids])
        input_prefix_mask = torch.LongTensor([input_prefix_mask])
        past_key_values = None

        generated_tokens = []
        new_id_pos = 0

        while len(generated_tokens) < max_generated_tokens:
            _, logits, past_key_values = model(
                input_ids=input_ids[:, new_id_pos:].to(self.device),
                input_prefix_mask=input_prefix_mask.to(self.device),
                predict_one_token=True,
                past_key_values=past_key_values,
            )

            next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).to("cpu").item()

            generated_tokens += [next_token]
            if next_token == eos_token_id:
                break

            new_id_pos = input_prefix_mask.shape[1]

            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]]).long(),
            ], dim=1)
            input_prefix_mask = torch.cat([
                input_prefix_mask,
                torch.zeros((input_prefix_mask.shape[0], 1)).long(),
            ], dim=1)

            yield process_response(tokenizer.decode(generated_tokens))

        return process_response(tokenizer.decode(generated_tokens))


def chat_template(history: list[tuple[str, str]], current: str):
    prompt = ""
    chat_round = 0
    for question, answer in history:
        prompt += f"[Round {chat_round}]\n问：{question}\n答：{answer}\n"
        chat_round += 1
    prompt += f"[Round {chat_round}]\n问：{current}\n答："
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
