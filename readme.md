
# chatglm-q

A [ChatGLM](https://huggingface.co/THUDM/chatglm-6b) reference implementation without Huggingface `transformers`. This implementation is optimized for ONNX export, int8 & int4 GPTQ quantization and more.

Currently [OpenAI Triton](https://github.com/openai/triton) only supports linux.

Install:
```bash
pip install git+https://github.com/K024/chatglm-q
# only available on linux + cuda
pip install triton
```

Usage:
```py
import torch
from chatglm_q.decoder import ChatGLMDecoder

device = torch.device("cuda")
decoder = ChatGLMDecoder.from_pretrained("K024/chatglm-6b-int4g32", device=device)

for text in decoder.generate("[Round 0]\n问：我是谁？\n答："):
    print(text)
```

Web UI:
```bash
pip install streamlit streamlit-chat
cd examples
streamlit run web-ui.py
```
