import re
import numpy
import base64
import unicodedata
import torch
from pathlib import Path
from typing import Any, Union, Literal
import tiktoken


class BatchEncoding(dict[str, torch.Tensor]):
    def to(self, device):
        for key in list(self.keys()):
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        return self

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError:
            raise AttributeError

    def __setattr__(self, item: str, value: Any):
        self[item] = value


PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<R>", "<S>", "<X>", "<mask>", "<sep>",
) + tuple([f"<extra_{i}>" for i in range(200)])


class QwenTokenizer:
    def __init__(self, vocab_file):
        assert vocab_file is not None
        self.vocab_file = vocab_file

        mergeable_ranks = self.mergeable_ranks = {
            base64.b64decode(token): int(rank)
            for line in Path(vocab_file).read_bytes().splitlines()
            for token, rank in [line.split()] if line
        }

        special_tokens = self.special_tokens = {
            tok: idx
            for idx, tok in enumerate(SPECIAL_TOKENS, len(mergeable_ranks))
        }

        self.text_tokenizer = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        self.bos_id = special_tokens["<|endoftext|>"]
        self.eos_id = special_tokens["<|endoftext|>"]
        self.unk_id = special_tokens["<|endoftext|>"]
        self.pad_id = special_tokens["<|endoftext|>"]
        self.im_start = special_tokens["<|im_start|>"]
        self.im_end = special_tokens["<|im_end|>"]


    def __len__(self):
        return len(self.mergeable_ranks) + len(self.special_tokens)

    def __getitem__(self, key: str):
        if key in self.mergeable_ranks:
            return self.mergeable_ranks[key]
        if key in self.special_tokens:
            return self.special_tokens[key]
        return self.unk_id

    def encode(
        self, text: str, add_special_tokens=True, allowed_special='all'
    ) -> list[int]:
        """
        text: Text to encode.
        add_special_tokens: No effect
        """
        text = unicodedata.normalize("NFC", text)
        tokens = self.text_tokenizer.encode(text, allowed_special=allowed_special)
        return tokens

    def decode(self, text_ids: list[int]) -> str:
        return self.text_tokenizer.decode(text_ids)

    def __call__(
        self,
        text: Union[str, list[str]],
        add_special_tokens = True,
        padding: Literal[True, False, "left", "right"] = False, # default pad to left
        max_length: int = None,
        return_tensors: Literal[False, "pt", "np"] = False,
        return_labels = False,
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]

        input_ids = [self.encode(t, add_special_tokens) for t in text]
        if max_length:
            input_ids = [i[:max_length] for i in input_ids]

        attention_mask = [[1] * len(i) for i in input_ids]
        position_ids = [list(range(len(i))) for i in input_ids]

        max_seq_length = max([len(i) for i in input_ids])
        pad_length = [max_seq_length - len(i) for i in input_ids]

        if padding == "right":
            input_ids = [i + p * [self.pad_id] for i, p in zip(input_ids, pad_length)]
            attention_mask = [a + p * [0] for a, p in zip(attention_mask, pad_length)]
            position_ids = [i + p * [0] for i, p in zip(position_ids, pad_length)]
        elif padding == "left" or padding == True:
            input_ids = [p * [self.pad_id] + i for i, p in zip(input_ids, pad_length)]
            attention_mask = [p * [0] + a for a, p in zip(attention_mask, pad_length)]
            position_ids = [p * [0] + i for i, p in zip(position_ids, pad_length)]
        else:
            assert not return_tensors, "set padding=True when return_tensors"

        if return_tensors == "np":
            input_ids = numpy.array(input_ids, dtype=numpy.int64)
            attention_mask = numpy.array(attention_mask, dtype=numpy.int64)
            position_ids = numpy.array(position_ids, dtype=numpy.int64)
        elif return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            position_ids = torch.tensor(position_ids, dtype=torch.long)

        inputs = BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if return_labels:
            assert return_tensors == "pt", "'return_labels' should be used with return_tensors='pt'"
            # -100: CrossEntropyLoss ignore_index
            labels = input_ids.masked_fill(~attention_mask.bool(), -100)
            inputs["labels"] = labels

        return inputs
