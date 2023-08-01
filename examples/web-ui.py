import torch
import streamlit as st
from chatglm_q.loader import from_pretrained
from chatglm_q.chatglm import ChatGLMDecoder


# page state

@st.cache_resource
def create_model() -> ChatGLMDecoder:
    device = torch.device("cuda")
    torch_dtype = torch.float16
    decoder = from_pretrained("K024/chatglm2-6b-int4g32", device, torch_dtype)
    # decoder.time_log = True # log generation performance
    return decoder

with st.spinner("加载模型中..."):
    model = create_model()


if "history" not in st.session_state:
    st.session_state["history"] = []


# parameters

with st.sidebar:
    st.markdown("## 采样参数")

    max_tokens = st.number_input("max_tokens", min_value=1, max_value=2000, value=800)
    temperature = st.number_input("temperature", min_value=0.1, max_value=4.0, value=1.0)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=50)

    kwargs = dict(max_generated_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)

    if st.button("清空上下文"):
        st.session_state.history = []

    st.markdown(f"""
    Source: [chatglm-q](https://github.com/K024/chatglm-q)

    Models: [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b) / [InternLM](https://github.com/InternLM/InternLM)

    Current:
    - Name: {model.model.__class__.__name__}
    - Quant: {model.config.quant_type}
    - Act: {model.config.torch_dtype}
    """)


# main body

st.markdown(f"## {model.model.__class__.__name__}")

history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("请在下方输入消息开始会话")


for idx, (question, answer) in enumerate(history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

question = st.chat_input("消息", key="message")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        empty = st.empty()
        with st.spinner("正在回复中"):
            for answer in model.chat(history, question, **kwargs):
                empty.write(answer)

    st.session_state.history = history + [(question, answer)]
