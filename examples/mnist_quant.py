# %%
import torch
from torch import nn

# PyTorch tutorial/quick start
# torch.save(next(iter(test_data_loader)), "test_data.pth")
data, labels = torch.load("test_data.pth")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# %%
from chatglm_q.int8.quantizer import GPTQLinearQuantizer, get_quant_int8_linear

qlayers: dict[str, GPTQLinearQuantizer] = {}

qmodel = NeuralNetwork()
qmodel.load_state_dict(torch.load("model.pth"))

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        qlayers[name] = GPTQLinearQuantizer(module)

# %%
model(data) # calibrate with data

for module in qlayers.values():
   module.remove_hook()

# %%
for name, module in qlayers.items():
    parent_path, module_name = name.rsplit(".")
    parent = qmodel.get_submodule(parent_path)
    # Compare with naive quantization
    # setattr(parent, module_name, get_quant_int8_linear(module.layer))
    setattr(parent, module_name, module.get_quantized_linear(pring_loss=True))

# %%
print("mean error:", ((qmodel(data) - model(data)) ** 2).mean())
print("different predictions:", (qmodel(data).argmax(-1) - model(data).argmax(-1)).bool().sum())

# %%
torch.onnx.export(
   model,
   f="model.onnx",
   args=(data,),
   input_names=["input"],
   output_names=["output"],
   dynamic_axes={ "input": { 0: "batch_size" } },
)
torch.onnx.export(
   qmodel,
   f="qmodel.onnx",
   args=(data,),
   input_names=["input"],
   output_names=["output"],
   dynamic_axes={ "input": { 0: "batch_size" } },
)

# %%
import torch
import onnxruntime

data, labels = torch.load("test_data.pth")
data = data.numpy()

qmodel = onnxruntime.InferenceSession("qmodel.onnx", providers=["CPUExecutionProvider"])
model = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# %%
q_out, = qmodel.run(["output"], { "input": data })
out, = model.run(["output"], { "input": data })

# %%
print("mean error:", ((q_out - out) ** 2).mean())
print("different predictions:", (q_out.argmax(-1) - out.argmax(-1)).astype(bool).sum())

# %%
