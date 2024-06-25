"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""
import gc

import streamlit as st
import torch
from multiModalRag.embedding import build_multiModal_retriever, build_parentChunk_retriever
from multiModalRag.rag import split_image_text_types
import matplotlib.pyplot as plt
import numpy as np
# from model_pool.langchainLLM import ChatGLM4_LLM, build_chat_history
from model_pool.test import ChatGLM4_LLM, build_chat_history
from model_pool.LLM import MiniCPM_Llama3_int4, ChatGLM, MiniCPM_Llama3
import base64
from PIL import Image
from io import BytesIO
from langchain_community.document_transformers import LongContextReorder
import time

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
def get_glm4(model_path, gpu_device, gen_kwargs):
    model = ChatGLM4_LLM(model_path, gpu_device, gen_kwargs)
    return model

@st.cache_resource
def get_glm3(model_path, max_memory_map):
    model = ChatGLM(model_path, max_memory_map)
    return model

@st.cache_resource
def get_cpm(model_path, max_memory_map):
    model = MiniCPM_Llama3(model_path, max_memory_map)
    return model

@st.cache_resource
def get_multiModal_retriever(dataBase_url, embed_model_path, token_size):
    retriever = build_multiModal_retriever(dataBase_url, embed_model_path, token_size)
    return retriever

@st.cache_resource
def get_parentChunk_retriever(dataBase_url,
                              embed_model_path,
                              parent_chunkSize=500,
                              child_chunkSize=200):

    retriever = build_parentChunk_retriever(dataBase_url,
                                            embed_model_path,
                                            parent_chunkSize,
                                            child_chunkSize)
    return retriever


def convert_image_from_base64(base64_str):
    # base64 to bytes
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

if __name__ == "__main__":
    # context path
    dataBase_url = './data/rag'
    embed_model_path = './model_pool/bge-m3'
    token_size = '500token'

    # init retriever
    retriever_multiModal = get_multiModal_retriever(dataBase_url, embed_model_path, token_size)
    retriever_parentChunk = get_parentChunk_retriever(dataBase_url,
                                                      embed_model_path)

    # cpm config
    cpm_path = "model_pool/MiniCPM-Llama3-V-2_5"
    cpm_gpu_device = 1
    max_memory_map = {cpm_gpu_device: "20GiB"}

    # Load cpm
    cpm = get_cpm(cpm_path, max_memory_map)

    # glm4 config
    glm4_path = "model_pool/glm-4-9b-chat"
    gen_kwargs = {"max_length": 2500}
    glm4_gpu_device = "cuda:0"
    chat_prompt_template = "RAG_CHATGLM_TEMPLATE"
    query_transfer_prompt_template = "QUERY_TRANSFORM_TEMPLATE"

    # Load glm4
    glm = get_glm4(glm4_path, glm4_gpu_device, gen_kwargs)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "past_key_values" not in st.session_state:
        st.session_state.past_key_values = None

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
                if message["rag_img_path"] is not None:
                    st.image(message["rag_img_path"], caption='图片来自西门子《S7-1200 入门指南》')


    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()

    question = st.chat_input("请输入您的问题")

    if question:
        input_placeholder.markdown(question)
        history = st.session_state.history
        past_key_values = st.session_state.past_key_values

        # question_trans = glm.query_transfer(question, history)
        question_trans = question
        print("====transfer question====: ", question_trans)

        # multimodal rag
        res_multiModal_with_score = retriever_multiModal.vectorstore.similarity_search_with_score(question_trans)
        res_multiModal_score = res_multiModal_with_score[0][1]
        res_multiModal_most_related_type = ""
        res_multiModal_rag_text = ""

        if res_multiModal_score < 0.6:
            res_multiModal = retriever_multiModal.invoke(question_trans)
            res_multiModal_reorder = LongContextReorder().transform_documents(res_multiModal)

            res_multiModal_img_text_dict = split_image_text_types(res_multiModal_reorder)
            res_multiModal_most_related_type = res_multiModal_img_text_dict["most_related_type"]
            print("====multiModal rag doc====:", res_multiModal_img_text_dict['texts'])


            if res_multiModal_most_related_type == "image":
                # cpm
                res_multiModal_img_base64 = res_multiModal_img_text_dict['images'][0]
                # print(res_multiModal_img_base64)
                res_multiModal_rag_image = convert_image_from_base64(res_multiModal_img_base64)
                res_multiModal_rag_text = cpm.chat(res_multiModal_rag_image, question_trans)
                print("====RAG texts from image====:", res_multiModal_rag_text)

            elif res_multiModal_most_related_type == "text":
                res_multiModal_rag_text = res_multiModal_img_text_dict['texts'][0]
                print("====RAG texts from documents====:", res_multiModal_rag_text)

        # parent chunk rag
        res_parentChunk = retriever_parentChunk.invoke(question_trans)
        if len(res_parentChunk) == 0:
            res_parentChunk = ""
        else:
            res_parentChunk = res_parentChunk[0].page_content
        print(f"====RAG text parent chunk====: {res_parentChunk} \n")

        prompt_context = str({"上下文信息": res_parentChunk + '\n' + res_multiModal_rag_text})
        print("====RAG final resutls====:", prompt_context)

        # Chat with glm

        # stream output
        # for response, history, past_key_values in glm.model.stream_chat(
        #         glm.tokenizer,
        #         question,
        #         prompt_text,
        #         history,
        #         past_key_values=past_key_values,
        #         max_length=max_length,
        #         top_p=top_p,
        #         temperature=temperature,
        #         return_past_key_values=True,
        # ):
        #
        #     message_placeholder.markdown(response)

        response = glm.invoke(question,
                              chat_history=history,
                              context=prompt_context,
                              prompt_template="RAG_CHATGLM_TEMPLATE",
                              stream=True
                              )

        message_placeholder.markdown(response)

        history = build_chat_history(history, question, response)

        if res_multiModal_most_related_type == "image":
            st.image(res_multiModal_rag_image, caption='图片来自西门子《S7-1200 入门指南》')
            history[-1]['rag_img_path'] = res_multiModal_rag_image
        else:
            history[-1]['rag_img_path'] = None

        print("history:", history)
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values

        # streamlit run web_demo_chatglm.py

        # query1: 详细解释s7-1200在一个完整的扫描周期中所执行的所有任务过程。
        # query2: 如何在step 7 basic中创建一个项目?
        # query3: 如何将电源连接到S7-1200?
        # query4: s7-1200数据类型中，字符char的大小和范围是多少？
