
# chatglm-q

A [ChatGLM](https://huggingface.co/THUDM/chatglm-6b) reference implementation without [Huggingface transformers](https://huggingface.co/docs/transformers). This implementation is optimized for ONNX export, int8 & int4 [GPTQ](https://github.com/IST-DASLab/gptq) quantization and more.

Currently [OpenAI Triton](https://github.com/openai/triton) only supports Linux. If your host system is Windows 11, WSL 2 with Cuda is also supported.

## Installation

Install PyTorch first. OpenAI Triton is also packed with PyTorch 2 Linux releases.

```bash
pip install git+https://github.com/K024/chatglm-q
```

For WSL 2 users, first check your WSL version. **DO NOT** install nvidia-driver in WSL 2. For more details, see [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```bat
> wsl --list -v
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

## Usage

```python
import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template

device = torch.device("cuda")
decoder = ChatGLMDecoder.from_pretrained("K024/chatglm-6b-int4g32", device=device)

prompt = chat_template([], "我是谁？")
for text in decoder.generate(prompt):
    print(text)
```

## Web UI

```bash
pip install streamlit streamlit-chat
cd examples
streamlit run web-ui.py
```

## Avaiable models

| Type      | Huggingface Hub                                                               |      |
| :-------- | :---------------------------------------------------------------------------- | :--- |
| int8      | [K024/chatglm-6b-int8](https://huggingface.co/K024/chatglm-6b-int8)           |      |
| int4g32   | [K024/chatglm-6b-int4g32](https://huggingface.co/K024/chatglm-6b-int4g32)     |      |
| onnx-u8s8 | [K024/ChatGLM-6b-onnx-u8s8](https://huggingface.co/K024/ChatGLM-6b-onnx-u8s8) |      |

The model weights are released under the same license as ChatGLM-6b, see [MODEL LICENSE](https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE).

ONNX model inference is not supported by this repo. The inference code is packed in the huggingface hub.

## TODO

- [ ] Integration with Huggingface Transformers
- [ ] Support cuda operator on windows
- [ ] LoRA, Bias Tuning and Prefix Tuning
