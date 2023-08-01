
# chatglm-q

一个仅供参考的 [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b) & [InternLM](https://github.com/InternLM/InternLM) 实现，去掉了 [Huggingface transformers](https://huggingface.co/docs/transformers) 依赖。这个实现为 ONNX 模型导出、int4 和 int8 量化等进行了优化调整。由于当前 [OpenAI Triton](https://github.com/openai/triton) 只支持 Linux，在 Windows 上，需要使用支持 Cuda 的 WSL2 运行（Win 11 或者 Win 10 21H2+）。

A [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b) & [InternLM](https://github.com/InternLM/InternLM) reference implementation without [Huggingface transformers](https://huggingface.co/docs/transformers). This implementation is optimized for ONNX export, int8 & int4 [GPTQ](https://github.com/IST-DASLab/gptq) quantization and more. Currently [OpenAI Triton](https://github.com/openai/triton) only supports Linux. If your host system is Windows 11 or Windows 10 21H2 and higher, WSL 2 with Cuda is also supported.

## Updates

新增 [InternLM](https://github.com/InternLM/InternLM) 基础支持，examples 中仍按照 ChatGLM2 给出示例代码，请自行修改部分代码。

本仓库已全面升级到 ChatGLM2，不再支持第一代 ChatGLM-6b，历史代码参考 [chatglm-legacy](https://github.com/K024/chatglm-q/tree/chatglm-legacy) 分支。

注：由于 ONNXRuntime MatMulInteger 算子问题，v2 模型的 ONNX 量化模型无法在 GPU 上运行，在 x86-64 的 CPU 上存在数值偏差，无法正常输出。

## Installation

首先需要安装依赖项 PyTorch。如果是 PyTorch 2.0 版本，其内置了 OpenAI Triton，否则需要额外安装 `triton` 包。这个包同时依赖 `build-essential` 构建工具。接下来可以通过下面的命令安装模块：

Install PyTorch first. OpenAI Triton is also packed with PyTorch 2 Linux releases. If you are using PyTorch 1.x, you should install `triton` manually. This package also requires build toolchain like `build-essential`. Then install this package with following command:

```bash
pip install --upgrade git+https://github.com/K024/chatglm-q
```

你也可以先克隆该仓库再使用 `pip install .` 来安装。

You can also clone this repo and install it with `pip install .`.

---

对于 WSL2 的用户，请先确认当前的 WSL 版本。**不要**在 WSL2 中手工安装 nvidia-driver，参考 [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

For WSL 2 users, first check your WSL version. **DO NOT** install nvidia-driver in WSL 2. For more details, see [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```bat
> wsl --list -v
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

## Usage

```python
import torch
from chatglm_q.loader import from_pretrained

device = torch.device("cuda")
# optionally pass a `torch_dtype=torch.float16` to set the activation dtype
decoder = from_pretrained("K024/chatglm2-6b-int4g32", device=device)

for text in decoder.chat([], prompt):
    print(text)
```

权重转换、模型量化、ONNX 模型导出等内容，请参考 [examples](./examples) 下的文件并作出必要的修改。

For more examples like weight conversion, manual quantization, onnx model export and more, please check out [examples](./examples) directory and make your own modifications.

## Web UI

```bash
pip install streamlit
cd examples
streamlit run web-ui.py
```

## Avaiable models

| Type      | Huggingface Hub                                                               | Recommended For                               |
| :-------- | :---------------------------------------------------------------------------- | :-------------------------------------------- |
| int8      | [K024/chatglm2-6b-int8](https://huggingface.co/K024/chatglm2-6b-int8)           | Linux/WSL2 CUDA 9G+ VRAM                      |
| int4g32   | [K024/chatglm2-6b-int4g32](https://huggingface.co/K024/chatglm2-6b-int4g32)     | Linux/WSL2 CUDA 6G+ VRAM                      |

模型权重按照原始模型相同的协议发布，见 [MODEL LICENSE](https://huggingface.co/THUDM/chatglm2-6b/blob/main/MODEL_LICENSE)。

The model weights are released under the same license as ChatGLM2-6b, see [MODEL LICENSE](https://huggingface.co/THUDM/chatglm2-6b/blob/main/MODEL_LICENSE).

## TODO

- [ ] Integration with Huggingface Transformers
- [ ] Support cuda operator on windows
- [ ] P-Tuning
