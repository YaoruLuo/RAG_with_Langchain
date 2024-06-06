"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""

import streamlit as st
import torch
from multiModalRag.embedding import build_retriever
from multiModalRag.rag import split_image_text_types
import matplotlib.pyplot as plt
import numpy as np
from chatApp.model_pool.LLM import ChatGLM, PROMPT_TEMPLATE
import base64
from PIL import Image
from io import BytesIO

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.set_page_config(
        page_title="Chat Demo Chatglm",
        page_icon=":robot:",
        layout="wide"
    )


st.title(":green[SIEMENS IOX-AI]")

@st.cache_resource
def get_LLM(model_path, max_memory_map):
    model = ChatGLM(model_path, max_memory_map)
    return model

@st.cache_resource
def get_retriever(text_fpath, img_fpath, embed_model_path):
    retriever = build_retriever(text_fpath, img_fpath, embed_model_path)
    return retriever

# @st.cache_resource
# def load_database(local_database_path, index_save_path):
#     # Document text extraction, split into chunks (e.g. sentences/paragraphs). Vector database.
#     print("Extracting documents...")
#     textractor = Textractor(paragraphs=True)
#
#     embeddings = Embeddings(content=True)
#     loadIndex(local_database_path, index_save_path, embeddings)
#
#     return embeddings


def convert_image_from_base64(base64_str):
    # 解码Base64字符串为二进制数据
    image_data = base64.b64decode(base64_str)

    # 使用PIL.Image从二进制数据创建图像对象
    image = Image.open(BytesIO(image_data))

    return image

if __name__ == "__main__":
    # context path
    text_fpath = './data/1200/doc/500token'
    img_fpath = './data/1200'
    embed_model_path = './model_pool/bge-m3'

    # LLM path
    LLM_path = "model_pool/chatglm3-6b"
    max_memory_map = {0: "11GB", 1: "11GB"}

    # txtai local database path
    # local_database_path = "./data/siemens"
    # index_save_path = "data/indexing_mini"

    # Load model
    LLM = get_LLM(LLM_path, max_memory_map)
    retriever = get_retriever(text_fpath, img_fpath, embed_model_path)
    # embeddings = load_database(local_database_path, index_save_path)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "past_key_values" not in st.session_state:
        st.session_state.past_key_values = None

    max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

    buttonClean = st.sidebar.button("清理会话历史", key="clean")
    if buttonClean:
        st.session_state.history = []
        st.session_state.past_key_values = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            with st.chat_message(name="user", avatar="user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message(name="assistant", avatar="assistant"):
                st.markdown(message["content"])
                st.image(message["rag_img_path"])


    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()


    question = st.chat_input("请输入您的问题")

    if question:
        input_placeholder.markdown(question)
        history = st.session_state.history
        past_key_values = st.session_state.past_key_values

        # rag
        docs = retriever.invoke(question, limit=6)
        img_text_dict = split_image_text_types(docs)
        print("question", question)
        print("doc:", img_text_dict['texts'])

        img_base64 = img_text_dict['images'][0]
        if len(img_text_dict['texts']) != 0:
            text = img_text_dict['texts'][0]
            prompt_text = PROMPT_TEMPLATE["RAG_CHATGLM_TEMPLATE"].format(question=question, text=text)
        else:
            prompt_text = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question=question)

        print(prompt_text)

        for response, history, past_key_values in LLM.model.stream_chat(
                LLM.tokenizer,
                question,
                prompt_text,
                history,
                past_key_values=past_key_values,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                return_past_key_values=True,
        ):
            message_placeholder.markdown(response)

        image = convert_image_from_base64(img_base64)
        st.image(image)
        history[-1]['rag_img_path'] = image

        print("history:", history)
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values


        # streamlit run web_demo_chatglm.py