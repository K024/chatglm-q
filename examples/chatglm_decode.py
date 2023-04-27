# %%
import torch
from chatglm_q.loader import load_state_dict, load_quant_model

# model = load_state_dict("../models/chatglm-6b-safe", dtype=torch.half)
model = load_quant_model("../models/chatglm-6b-int8.safetensors")

# %%

device = torch.device("cuda")
model = model.to(device)

# %%
from chatglm_q.tokenizer import ChatGLMTokenizer

tokenizer = ChatGLMTokenizer("../models/chatglm-6b-safe/sentencepiece.model")

# %%
from chatglm_q.decoder import ChatGLMDecoder

decoder = ChatGLMDecoder(model, tokenizer, device=device)

# %%
for text in decoder.generate("[Round 0]\n问：我是谁？\n答：", tokenizer["<eop>"]):
    print(text)

# %%
