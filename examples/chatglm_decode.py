# %%
import torch
from chatglm_q.decoder import ChatGLMDecoder

device = torch.device("cuda")
decoder = ChatGLMDecoder.from_pretrained("../models/chatglm-6b-int4g32", device=device)

# %%
for text in decoder.generate("[Round 0]\n问：我是谁？\n答："):
    print(text)

# %%
