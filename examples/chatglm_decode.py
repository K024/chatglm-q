# %%
import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template

device = torch.device("cuda")
decoder = ChatGLMDecoder.from_pretrained("../models/chatglm-6b-int4g32", device=device)

# %%
prompt = chat_template([], "我是谁？")
for text in decoder.generate(prompt):
    print(text)

# %%
