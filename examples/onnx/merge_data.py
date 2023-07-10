# %%
from tqdm.auto import tqdm
from pathlib import Path

file_dir = Path(__file__).parent
tokenizer_path = file_dir / "../../models/chatglm-6b-safe/sentencepiece.model"

export_path = file_dir / "../../models/chatglm-6b-int8-onnx/chatglm-6b-int8-opt.onnx"
export_path.parent.mkdir(exist_ok=True)
export_path = str(export_path.absolute())

# %%
import onnx
import onnx.external_data_helper

model = onnx.load(export_path)
onnx.external_data_helper.convert_model_from_external_data(model)

# %%
tensors = list(onnx.external_data_helper._get_initializer_tensors(model))

# %%
size_threshold = 256
current_sum = 0

bar = tqdm(tensors)
for tensor in bar:
  size = onnx.external_data_helper.sys.getsizeof(tensor.raw_data)
  if (tensor.HasField("raw_data") and size >= size_threshold):
    if any(d > 50_000 for d in tensor.dims):
      file_name = "model_weights_0.bin"
    else:
      current_sum += size
      file_idx = current_sum // 1_000_000_000 + 1
      file_name = f"model_weights_{file_idx}.bin"
      bar.set_postfix_str(f"{file_idx=}")
    onnx.external_data_helper.set_external_data(tensor, file_name)

# %%
save_path = file_dir / "../../models/chatglm-6b-int8-onnx-merged/chatglm-6b-int8.onnx"
save_path.parent.mkdir(exist_ok=True)
save_path = str(save_path.absolute())
onnx.save_model(model, save_path)

# %%
