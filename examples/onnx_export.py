# %%
from pathlib import Path

file_dir = Path(__file__).parent
model_path = file_dir / "../models/chatglm-6b-int8/"
tokenizer_path = file_dir / "../models/chatglm-6b-safe/sentencepiece.model"

export_path = file_dir / "../models/chatglm-6b-int8-onnx/chatglm-6b-int8.onnx"
export_path.parent.mkdir(exist_ok=True)
export_path = str(export_path.absolute())

# %%
import torch
from chatglm_q.loader import load_model_and_tokenizer

_, model, tokenizer = load_model_and_tokenizer(model_path)

input_ids, prefix_mask = tokenizer.encode("[Round 0]\n", "问：")

_, _, past_key_values = model(
    input_ids=torch.LongTensor([input_ids]),
    input_prefix_mask=torch.LongTensor([prefix_mask])
)

# %%
input_ids = [tokenizer["<n>"]]
prefix_mask = prefix_mask + [0]

input_args = (
    torch.LongTensor([input_ids]),
    None,
    torch.LongTensor([prefix_mask]),
    None,
    past_key_values,
    torch.tensor(True),
)

input_names = ["input_ids", "prefix_mask"]
output_names = ["logits"]
dynamic_axes = { 
    "input_ids": { 0: "batch_size", 1: "new_seq_length" },
    "prefix_mask": { 0: "batch_size", 1: "seq_length" },
}

for layer_idx in range(model.config.num_layers):
    input_names += [f"past_key_{layer_idx}", f"past_value_{layer_idx}"]
    output_names += [f"present_key_{layer_idx}", f"present_value_{layer_idx}"]

    dynamic_axes.update({
        f"past_key_{layer_idx}": { 0: "batch_size", 1: "past_seq_length" },
        f"past_value_{layer_idx}": { 0: "batch_size", 1: "past_seq_length" },
    })

input_names += ["use_past"]

# %%
torch.onnx.export(
    model,
    f=export_path,
    args=input_args,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
)

# %%
from onnxruntime.tools.optimize_onnx_model import optimize_model

output_path = file_dir / "../models/chatglm-6b-int8-onnx/chatglm-6b-int8-opt.onnx"
output_path.parent.mkdir(exist_ok=True)

optimize_model(Path(export_path), output_path)

# %%
