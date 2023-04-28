import torch
from .model import ChatGLMModel
from .tokenizer import ChatGLMTokenizer


def top_p_sampling(logits: torch.Tensor, top_k=100, top_p=0.8):
    # top_k
    probs = torch.softmax(logits.float(), dim=-1)
    probs, indices = torch.sort(probs, dim=-1, descending=True)
    probs = probs[..., :top_k]
    indices = indices[..., :top_k]

    # top_p
    cumsum = torch.cumsum(probs, dim=-1)
    probs[(cumsum - probs) > top_p] = 0.0
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # sample
    next_token = torch.multinomial(probs, num_samples=1)
    output = torch.gather(indices, dim=1, index=next_token)
    return output


class ChatGLMDecoder():
    def __init__(self, model: ChatGLMModel, tokenizer: ChatGLMTokenizer, top_k = 50, top_p = 0.8, device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.top_k = top_k
        self.device = device

    @torch.no_grad()
    def generate(self, prefix_text: str, eos_token_id: int, max_length = 200):
        model, tokenizer, top_k, top_p = self.model, self.tokenizer, self.top_k, self.top_p

        input_ids, input_prefix_mask = tokenizer.encode(prefix_text)
        input_ids = torch.LongTensor([input_ids])
        input_prefix_mask = torch.LongTensor([input_prefix_mask])
        past_key_values = None

        output_text = ""
        new_id_pos = 0

        while len(input_ids) < max_length:
            _, logits, past_key_values = model(
                input_ids=input_ids[:, new_id_pos:].to(self.device),
                input_prefix_mask=input_prefix_mask.to(self.device),
                predict_one_token=True,
                past_key_values=past_key_values,
            )

            next_token = top_p_sampling(logits[:, -1], top_k, top_p).to("cpu")

            output_text += tokenizer.decode(next_token[0].tolist())
            if next_token[0, 0].item() == eos_token_id:
                break

            yield output_text

            new_id_pos = input_prefix_mask.shape[1]

            input_ids = torch.cat([input_ids, next_token], dim=1)
            input_prefix_mask = torch.cat([
                input_prefix_mask,
                torch.zeros((input_prefix_mask.shape[0], 1)).long(),
            ], dim=1)

        return output_text
