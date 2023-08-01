from .decoder import InternLMDecoder, chat_template
from .model import InternLMConfig, InternLMModel
from .tokenizer import InternLMTokenizer


config_class = InternLMConfig
model_class = InternLMModel
tokenizer_class = InternLMTokenizer
decoder_class = InternLMDecoder


def create_int8_model(config=InternLMConfig(), dtype=None):
    try:
        from . import model as modeling
        from ..int8.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return InternLMModel(config, dtype)
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


def create_int4_model(config=InternLMConfig(), group_size=32, dtype=None):
    try:
        from . import model as modeling
        from ..int4 import qlinear
        from ..int4.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_group_size = qlinear.DEFAULT_GROUP_SIZE
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        qlinear.DEFAULT_GROUP_SIZE = group_size
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return InternLMModel(config, dtype)
    finally:
        qlinear.DEFAULT_GROUP_SIZE = prev_group_size
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding
