# %%
from tqdm.auto import tqdm
from pathlib import Path

export_path = Path("../../models/chatglm2-6b-int8-onnx/chatglm2-6b-int8-opt.onnx")
assert export_path.exists()
export_path = str(export_path.absolute())

# %%
import onnx
import onnx.external_data_helper

model = onnx.load(export_path)
onnx.external_data_helper.convert_model_from_external_data(model)

# %%
tensors = list(onnx.external_data_helper._get_initializer_tensors(model))

# %%
size_threshold = 128
current_sum = 0

bar = tqdm(tensors)
for tensor in bar:
  size = onnx.external_data_helper.sys.getsizeof(tensor.raw_data)
  if (tensor.HasField("raw_data") and size >= size_threshold):
    current_sum += size
    file_idx = current_sum // (2 * 1024 ** 3) + 1
    file_name = f"model_weights_{file_idx}.bin"
    bar.set_postfix_str(f"{file_idx=}")
    onnx.external_data_helper.set_external_data(tensor, file_name)

# %%
save_path = Path("../../models/chatglm2-6b-int8-onnx-merged/chatglm2-6b-int8.onnx")
save_path.parent.mkdir()
save_path = str(save_path.absolute())
onnx.save_model(model, save_path)

# %%
import shutil

tokenizer = "../../models/chatglm2-6b-int8/sentencepiece.model"
target = "../../models/chatglm2-6b-int8-onnx-merged/sentencepiece.model"

shutil.copy(tokenizer, target)
shutil.rmtree("../../models/chatglm2-6b-int8-onnx")
shutil.move("../../models/chatglm2-6b-int8-onnx-merged", "../../models/chatglm2-6b-int8-onnx")
