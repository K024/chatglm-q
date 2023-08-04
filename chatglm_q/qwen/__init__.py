from .decoder import QwenDecoder, chat_template
from .model import QwenConfig, QwenModel
from .tokenizer import QwenTokenizer


config_class = QwenConfig
model_class = QwenModel
tokenizer_class = QwenTokenizer
decoder_class = QwenDecoder


def create_int8_model(config=QwenConfig(), dtype=None):
    try:
        from . import model as modeling
        from ..int8.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return QwenModel(config, dtype)
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


def create_int4_model(config=QwenConfig(), group_size=32, dtype=None):
    try:
        from . import model as modeling
        from ..int4 import qlinear
        from ..int4.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_group_size = qlinear.DEFAULT_GROUP_SIZE
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        qlinear.DEFAULT_GROUP_SIZE = group_size
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return QwenModel(config, dtype)
    finally:
        qlinear.DEFAULT_GROUP_SIZE = prev_group_size
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding
