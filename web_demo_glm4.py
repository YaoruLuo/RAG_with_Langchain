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
import matplotlib.pyplot as plt
import numpy as np
from model_pool.langchainLLM import ChatGLM4_LLM, build_chat_history


arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.set_page_config(
        page_title="Chat Demo glm4",
        page_icon=":robot:",
        layout="wide"
    )


st.title(":green[GLM4-9b-Chat]")

@st.cache_resource
def get_glm4(model_path, gpu_device, gen_kwargs):
    model = ChatGLM4_LLM(model_path, gpu_device, gen_kwargs)
    return model



if __name__ == "__main__":
    # context path

    # glm4 config
    glm4_path = "model_pool/glm-4-9b-chat"
    gen_kwargs = {"max_length": 2500}
    glm4_gpu_device = "cuda:3"
    chat_prompt_template = "SIMPLE_CHAT"

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

    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()

    question = st.chat_input("请输入您的问题")

    if question:
        input_placeholder.markdown(question)
        history = st.session_state.history
        past_key_values = st.session_state.past_key_values
        print("====question====: ", question)

        # Chat with glm
        response = glm.invoke(question,
                              chat_history=history,
                              context=question,
                              prompt_template=chat_prompt_template,
                              )

        message_placeholder.markdown(response)

        history = build_chat_history(history, question, response)

        print("history:", history)
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values

        # streamlit run web_demo_chatglm.py

        # query1: 详细解释s7-1200在一个完整的扫描周期中所执行的所有任务过程。
        # query2: 如何在step 7 basic中创建一个项目?
        # query3: 如何将电源连接到S7-1200?
        # query4: s7-1200数据类型中，字符char的大小和范围是多少？
