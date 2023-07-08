import re
import numpy
import torch
from typing import Any, Union, Literal
from sentencepiece import SentencePieceProcessor


class BatchEncoding(dict):
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


class ChatGLM2Tokenizer:
    def __init__(self, vocab_file):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>"]
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))
        self.vocab_size = len(self.text_tokenizer) + len(self.special_tokens)
        self.true_vocab_size = len(self.text_tokenizer)
        
        self.bos_id: int = self.text_tokenizer.bos_id()
        self.eos_id: int = self.text_tokenizer.eos_id()
        self.pad_id: int = self.text_tokenizer.unk_id()

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, key: str):
        if key in self.special_tokens:
            return len(self.text_tokenizer) + self.special_tokens.index(key)
        return self.text_tokenizer[key]

    def encode(
        self, text: str, text_pair: str = None, add_special_tokens=True,
    ) -> list[int]:
        """
        text: Text to encode.
        text_pair: Expected answer to encode.
        add_special_tokens: Add "[gMASK]" "<sop>" before `text` and "</s>" after `text_pair`
        """
        tokens = self.text_tokenizer.encode(text)
        if add_special_tokens:
            tokens = [self["[gMASK]"], self["<sop>"]] + tokens

        if text_pair is not None:
            pair_tokens = self.text_tokenizer.encode(text_pair)
            tokens += pair_tokens
            if add_special_tokens:
                tokens += [self.eos_id]

        return tokens

    def decode(self, text_ids: list[int]) -> str:
        text_ids = list(filter(lambda x: x < self.true_vocab_size, text_ids))
        text = self.text_tokenizer.decode(text_ids)
        return text

    def __call__(
        self,
        text: Union[str, list[str]],
        text_pair: Union[str, list[str]] = None,
        add_special_tokens = True,
        padding: Literal[True, False, "left", "right"] = False, # default pad to left
        max_length: int = None,
        return_tensors: Literal[False, "pt", "np"] = False,
    ) -> Any:
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        if text_pair is None:
            text_pair = [None] * len(text) 
        assert len(text) == len(text_pair)

        input_ids = []
        for t, tp in zip(text, text_pair):
            input_ids.append(self.encode(t, tp, add_special_tokens))

        attention_mask = []
        for inputs in input_ids:
            attention_mask.append([1] * len(inputs))

        position_ids = []
        for inputs in input_ids:
            position_ids.append(list(range(len(inputs))))

        if max_length:
            for i in range(len(input_ids)):
                input_ids[i] = input_ids[i][:max_length]
                attention_mask[i] = attention_mask[i][:max_length]
                position_ids[i] = position_ids[i][:max_length]

        max_seq_length = max(map(lambda x: len(x), input_ids))
        if padding == "right":
            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = input_ids[i] + pad_length * [self.pad_id]
                attention_mask[i] = attention_mask[i] + pad_length * [0]
                position_ids[i] = position_ids[i] + pad_length * [0]
        elif padding == "left" or padding == True:
            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = pad_length * [self.pad_id] + input_ids[i]
                attention_mask[i] = pad_length * [0] + attention_mask[i]
                position_ids[i] = pad_length * [0] + position_ids[i]
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

        return BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
