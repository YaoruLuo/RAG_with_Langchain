from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import base64
import os
from PIL import Image
from chatApp.model_pool.LLM import MiniCPM_Llama3_int4, ChatGLM

import pickle
import json
from langchain_text_splitters import CharacterTextSplitter


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
Generate texts, images, tables summaries.
"""

# Generate summaries of text elements
def generate_text_summaries(texts):
    """
    Summarize text elements
    """

    # Load model
    model_dir = "../model_pool/chatglm3-6b"
    max_memory_map = {0: "11GB", 1: "11GB"}
    prompt_template = "SUMMARY_CHATGLM_TEMPLATE"
    model = ChatGLM(model_dir, max_memory_map)

    # Initialize empty summaries
    text_summaries = []

    for text in texts[:3]:
        response, _ = model.chat(question=text, history=[], prompt_template=prompt_template)
        print(f"text: {text}\nsummary:{response}")
        text_summaries.append(response)

    return text_summaries

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries_list = []

    # Store image paths
    image_path_list= []

    prompt = """以西门子s7-1200 PLC为背景，详细总结图中的信息。"""

    model = MiniCPM_Llama3_int4(model_path="../model_pool/MiniCPM-Llama3-V-2_5-int4")

    msgs = [{'role': 'user', 'content': prompt}]

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".png"):
            img_path = os.path.join(path, img_file)
            image = Image.open(img_path).convert('RGB')
            image_path_list.append(img_path)

            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)

            summerize_txt = model.chat(image, msgs)
            image_summaries_list.append(summerize_txt)

    return img_base64_list, image_summaries_list, image_path_list

def save_to_json(image_urls, summary, filename):
    if len(image_urls) != len(summary):
        raise ValueError("图片地址列表和图片列表的长度必须一致")

    data = {}

    for i in range(len(image_urls)):
        data[image_urls[i]] = summary[i]

    json_data = json.dumps(data, indent=4, ensure_ascii=False)

    with open(filename, 'w') as file:
        file.write(json_data)

    return json_data


if __name__ == "__main__":
    # Get text, table summaries
    text_path = '../data/1200/doc/s7-1200文档.txt'
    with open(text_path, "r") as f:
        texts = f.read()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=25
    )
    texts_token = text_splitter.split_text(texts)
    print("texts_token:", texts_token)

    with open('../data/1200/doc/250token/text.pkl', 'wb') as f:
        pickle.dump(texts_token, f)

    text_summaries = generate_text_summaries(texts_token)

    with open('../data/1200/doc/250token/summary.pkl', 'wb') as f:
        pickle.dump(text_summaries, f)

    # ==============================
    # # Get image summaries
    # img_path = "../data/1200/figures_manual"
    # img_base64_list, image_summaries_list, image_path_list = generate_img_summaries(img_path)
    #
    # # save img_base64_list
    # save_to_pke_path = '../data/1200/img_base64_list.pkl'
    # with open(save_to_pke_path, 'wb') as f:
    #     pickle.dump(img_base64_list, f)
    #
    # # save image_path_list, image_summaries_list
    # save_to_json_path = "../data/1200/path_summaries_list.json"
    # res = save_to_json(image_path_list, image_summaries_list, save_to_json_path)
