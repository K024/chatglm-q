# %%
from pathlib import Path

model_path = Path("../../models/chatglm2-6b-int8/")
export_path = Path("../../models/chatglm2-6b-int8-onnx/chatglm2-6b-int8.onnx")
export_path.parent.mkdir()
export_path = str(export_path.absolute())

# %%
import torch
from chatglm_q.chatglm import model as modeling
from chatglm_q.loader import load_model_and_tokenizer

modeling.ROTARY_VIEW_AS_COMPLEX = False

_, model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype=torch.float32)
inputs = tokenizer("[Round 0]\n\n问：", padding=True, return_tensors="pt")
_, _, past_key_values = model(**inputs)

# %%
inputs.input_ids = torch.LongTensor([[tokenizer.pad_id]])
inputs.attention_mask = torch.cat([
    inputs.attention_mask,
    torch.LongTensor([[1]]),
], dim=1)
inputs.position_ids = inputs.position_ids[:, -1:] + 1

input_args = (
    inputs.input_ids,
    None, # input_embeddings
    inputs.attention_mask,
    inputs.position_ids,
    None, # labels
    past_key_values,
)

input_names = ["input_ids", "attention_mask", "position_ids"]
output_names = ["logits"]
dynamic_axes = { 
    "input_ids": { 0: "batch_size", 1: "new_seq_length" },
    "attention_mask": { 0: "batch_size", 1: "all_seq_length" },
    "position_ids": { 0: "batch_size", 1: "new_seq_length" },
}

for layer_idx in range(model.config.num_layers):
    input_names += [f"past_key_{layer_idx}", f"past_value_{layer_idx}"]
    output_names += [f"present_key_{layer_idx}", f"present_value_{layer_idx}"]

    dynamic_axes.update({
        f"past_key_{layer_idx}": { 0: "batch_size", 1: "past_seq_length" },
        f"past_value_{layer_idx}": { 0: "batch_size", 1: "past_seq_length" },
    })

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

output_path = "../../models/chatglm2-6b-int8-onnx/chatglm2-6b-int8-opt.onnx"
optimize_model(Path(export_path), Path(output_path))

# %%
