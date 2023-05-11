import re
from sentencepiece import SentencePieceProcessor


def replace_spaces_with_blank(match: re.Match[str]):
    return f"<|blank_{len(match.group())}|>"


def replace_blank_with_spaces(match: re.Match[str]):
    return " " * int(match.group(1))


class ChatGLMTokenizer:
    def __init__(self, vocab_file):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))

    def __len__(self):
        return len(self.text_tokenizer)

    def __getitem__(self, key: str):
        return self.text_tokenizer[key]

    def preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = text.replace("\t", "<|tab|>")
            text = re.sub(r" {2,80}", replace_spaces_with_blank, text)
        return text

    def encode(
        self, text: str, text_pair: str = None,
        linebreak=True, whitespaces=True,
        add_dummy_prefix=True, special_tokens=True,
    ) -> tuple[list[int], list[int]]:
        """
        text: Text to encode. Bidirectional part with a [gMASK] and an <sop> for causal LM.
        text_pair: causal LM part.
        linebreak: Whether to encode newline (\n) in text.
        whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self.preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text

        tokens = self.text_tokenizer.encode(text)
        prefix_mask = [1] * len(tokens)
        if special_tokens:
            tokens += [self.text_tokenizer["[gMASK]"], self.text_tokenizer["<sop>"]]
            prefix_mask += [1, 0]

        if text_pair is not None:
            text_pair = self.preprocess(text_pair, linebreak, whitespaces)
            pair_tokens = self.text_tokenizer.encode(text_pair)
            tokens += pair_tokens
            prefix_mask += [0] * len(pair_tokens)
            if special_tokens:
                tokens += [self.text_tokenizer["<eop>"]]
                prefix_mask += [0]

        return (tokens if add_dummy_prefix else tokens[2:]), prefix_mask

    def decode(self, text_ids: list[int]) -> str:
        text = self.text_tokenizer.decode(text_ids)
        text = text.replace("<n>", "\n")
        text = text.replace("<|tab|>", "\t")
        text = re.sub(r"<\|blank_(\d\d?)\|>", replace_blank_with_spaces, text)
        return text

    # TODO: __call__
