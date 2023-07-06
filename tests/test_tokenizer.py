# %%
import unittest
from pathlib import Path
from chatglm_q.tokenizer import ChatGLM2Tokenizer

file_dir = Path(__file__).parent
model_path = file_dir / "../models/chatglm2-6b-safe"
tokenizer_path = model_path / "sentencepiece.model"

# %%
class ChatGLMTokenizerTests(unittest.TestCase):

    def test_tokenize(self):
        tokenizer = ChatGLM2Tokenizer(tokenizer_path)

        input_ids = tokenizer.encode("[Round 0]\n问：\t", "问题1")

        self.assertEqual(
            input_ids,
            [64790, 64792, 790, 30951, 517, 30910, 30940, 30996, 13, 54761, 31211, 12, 30910, 31639, 30939, 2],
        )

        self.assertEqual(
            [tokenizer["[gMASK]"], tokenizer["<sop>"], tokenizer['<eop>'], tokenizer['</s>']],
            [64790, 64792, 64793, 2],
        )

    def test_blank_and_tab(self):
        tokenizer = ChatGLM2Tokenizer(tokenizer_path)

        input_ids = tokenizer.encode("\t   \t     \n \n", add_special_tokens=False)

        self.assertEqual(
            input_ids,
            [30910, 12, 2951, 12, 3729, 13, 30910, 13],
        )

        decoded = tokenizer.decode(input_ids)

        self.assertEqual(
            decoded,
            "\t   \t     \n \n",
        )


if __name__ == '__main__':
    unittest.main()
