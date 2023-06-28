
# chatglm-q

A [ChatGLM](https://huggingface.co/THUDM/chatglm-6b) reference implementation without [Huggingface transformers](https://huggingface.co/docs/transformers). This implementation is optimized for ONNX export, int8 & int4 [GPTQ](https://github.com/IST-DASLab/gptq) quantization and more.

Currently [OpenAI Triton](https://github.com/openai/triton) only supports Linux. If your host system is Windows 11 or Windows 10 21H2 and higher, WSL 2 with Cuda is also supported.

一个仅供参考的 [ChatGLM](https://huggingface.co/THUDM/chatglm-6b) 实现，去掉了 [Huggingface transformers](https://huggingface.co/docs/transformers) 依赖。这个实现为 ONNX 模型导出、int4 和 int8 量化等进行了优化调整。由于当前 [OpenAI Triton](https://github.com/openai/triton) 只支持 Linux，在 Windows 上，需要使用支持 Cuda 的 WSL2 运行（Win 11 或者 Win 10 21H2+）。

## Installation

Install PyTorch first. OpenAI Triton is also packed with PyTorch 2 Linux releases. If you are using PyTorch 1.x, you should install `triton` manually. This package also requires build toolchain like `build-essential`. Then install this package with following command:

首先需要安装依赖项 PyTorch。如果是 PyTorch 2.0 版本，其内置了 OpenAI Triton，否则需要额外安装 `triton` 包。这个包同时依赖如 `build-essential` 的构建工具。接下来可以通过下面的命令安装模块：

```bash
pip install git+https://github.com/K024/chatglm-q
```

You can also clone this repo and install it with `pip install .`.

你也可以先克隆该仓库再使用 `pip install .` 来安装。

---

For WSL 2 users, first check your WSL version. **DO NOT** install nvidia-driver in WSL 2. For more details, see [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

对于 WSL2 的用户，请先确认当前的 WSL 版本。**不要**在 WSL2 中手工安装 nvidia-driver，参考 [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

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
# optionally pass a `torch_dtype=torch.float16` to set the activation dtype
decoder = ChatGLMDecoder.from_pretrained("K024/chatglm-6b-int4g32", device=device)

prompt = chat_template([], "我是谁？")
for text in decoder.generate(prompt):
    print(text)
```

For more examples like weight conversion, manual quantization, onnx model export and more, please check out [examples](./examples) directory and make your own modifications.

权重转换、模型量化、ONNX 模型导出等内容，请参考 [examples](./examples) 下的文件并作出必要的修改。

## Web UI

```bash
pip install streamlit streamlit-chat
cd examples
streamlit run web-ui.py
```

## Avaiable models

| Type      | Huggingface Hub                                                               | Recommended For                               |
| :-------- | :---------------------------------------------------------------------------- | :-------------------------------------------- |
| int8      | [K024/chatglm-6b-int8](https://huggingface.co/K024/chatglm-6b-int8)           | Linux/WSL2 CUDA 9G+ VRAM                      |
| int4g32   | [K024/chatglm-6b-int4g32](https://huggingface.co/K024/chatglm-6b-int4g32)     | Linux/WSL2 CUDA 6G+ VRAM                      |
| onnx-u8s8 | [K024/ChatGLM-6b-onnx-u8s8](https://huggingface.co/K024/ChatGLM-6b-onnx-u8s8) | x86-64 with AVX2/AVX512 / ARM64 (Apple M1/M2) |

The model weights are released under the same license as ChatGLM-6b, see [MODEL LICENSE](https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE).

ONNX model inference is not supported by this repo. The inference code is packed in the huggingface hub.

模型权重按照原始模型相同的协议发布，见 [MODEL LICENSE](https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE)。

ONNX 模型的推理代码和权重文件一同打包在 huggingface hub 中。

## TODO

- [ ] Integration with Huggingface Transformers
- [ ] Support cuda operator on windows
- [ ] LoRA, Bias Tuning and Prefix Tuning
