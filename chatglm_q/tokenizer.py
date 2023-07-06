import re
from sentencepiece import SentencePieceProcessor


class ChatGLM2Tokenizer:
    def __init__(self, vocab_file):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>"]
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))
        self.vocab_size = len(self.text_tokenizer) + len(self.special_tokens)
        self.true_vocab_size = len(self.text_tokenizer)

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
                tokens += [self["</s>"]]

        return tokens

    def decode(self, text_ids: list[int]) -> str:
        text_ids = list(filter(lambda x: x < self.true_vocab_size, text_ids))
        text = self.text_tokenizer.decode(text_ids)
        return text

    # TODO: __call__
