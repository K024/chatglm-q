import re
from .decoder import ChatGLMDecoder, chat_template
from .model import ChatGLM2Config, ChatGLM2Model
from .tokenizer import ChatGLM2Tokenizer


config_class = ChatGLM2Config
model_class = ChatGLM2Model
tokenizer_class = ChatGLM2Tokenizer
decoder_class = ChatGLMDecoder


def create_int8_model(config=ChatGLM2Config(), dtype=None):
    try:
        from . import model as modeling
        from ..int8.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLM2Model(config, dtype)
    finally:
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding


def create_int4_model(config=ChatGLM2Config(), group_size=32, dtype=None):
    try:
        from . import model as modeling
        from ..int4 import qlinear
        from ..int4.qlinear import DynamicQuantizeLinear, QEmbedding
        prev_group_size = qlinear.DEFAULT_GROUP_SIZE
        prev_linear, prev_embedding = modeling.Linear, modeling.Embedding
        qlinear.DEFAULT_GROUP_SIZE = group_size
        modeling.Linear, modeling.Embedding = DynamicQuantizeLinear, QEmbedding

        return ChatGLM2Model(config, dtype)
    finally:
        qlinear.DEFAULT_GROUP_SIZE = prev_group_size
        modeling.Linear, modeling.Embedding = prev_linear, prev_embedding
