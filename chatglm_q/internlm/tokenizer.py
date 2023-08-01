import re
import numpy
import torch
from typing import Any, Union, Literal
from sentencepiece import SentencePieceProcessor


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


class InternLMTokenizer:
    def __init__(self, vocab_file):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))
        self.bos_id = self["<s>"]
        self.eos_id = self["</s>"]
        self.unk_id = self["<unk>"]
        self.pad_id = self["</s>"]
        self.eoh_id = self["<eoh>"]
        self.eoa_id = self["<eoa>"]

    def __len__(self):
        return len(self.text_tokenizer)

    def __getitem__(self, key: str):
        return self.text_tokenizer[key]

    def encode(
        self, text: str, add_special_tokens=True,
    ) -> list[int]:
        """
        text: Text to encode.
        add_special_tokens: Add "<s>" as first token
        """
        tokens = self.text_tokenizer.encode(text)
        if add_special_tokens:
            tokens = [self["<s>"]] + tokens
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
