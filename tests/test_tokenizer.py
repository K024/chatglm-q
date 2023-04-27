# %%
import unittest
from pathlib import Path
from chatglm_q.tokenizer import ChatGLMTokenizer

file_dir = Path(__file__).parent
model_path = file_dir / "../models/chatglm-6b-safe"
tokenizer_path = model_path / "sentencepiece.model"

# %%
class ChatGLMTokenizerTests(unittest.TestCase):

    def test_tokenize(self):
        tokenizer = ChatGLMTokenizer(tokenizer_path)

        input_ids, prefix_mask = tokenizer.encode("[Round 0]\n问：\t", "问题1")

        self.assertEqual(
            input_ids,
            [53, 6945, 5, 8, 42, 4, 64286, 12, 130008, 130001, 130004, 5, 63963, 9, 130005],
        )

        self.assertEqual(
            prefix_mask,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        )

        self.assertEqual(
            [tokenizer["[gMASK]"], tokenizer["<sop>"], tokenizer['<eop>']],
            [130001, 130004, 130005],
        )


    def test_blank_and_tab(self):
        tokenizer = ChatGLMTokenizer(tokenizer_path)

        input_ids, _ = tokenizer.encode("\t   \t     \n \n", special_tokens=False)

        self.assertEqual(
            input_ids,
            [5, 130008, 130010, 130008, 130012, 4, 5, 4],
        )

        self.assertEqual(
            [tokenizer["<|blank_3|>"], tokenizer["<|blank_5|>"], tokenizer['<|tab|>'], tokenizer["<n>"]],
            [130010, 130012, 130008, 4],
        )

        decoded = tokenizer.decode(input_ids)

        self.assertEqual(
            decoded,
            "\t   \t     \n \n",
        )


if __name__ == '__main__':
    unittest.main()
