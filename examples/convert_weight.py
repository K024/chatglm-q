# %%
from huggingface_hub import snapshot_download
from chatglm_q.loader import convert_transformers_weights

# %%
path_or_repo_id = "https://huggingface.co/THUDM/chatglm2-6b"
cache_dir = ".cache"
token = None

model_path = snapshot_download(path_or_repo_id, cache_dir=cache_dir, token=token)

# %%
target_path = "./models/chatglm2-6b-safe"

convert_transformers_weights(model_path, target_path)
