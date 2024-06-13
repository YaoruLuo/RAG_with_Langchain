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
Generate texts, images summaries.
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

    for text in texts:
        response, _ = model.chat(question=text, history=[], prompt_template=prompt_template)
        print(f"text: {text}\nsummary:{response}")
        text_summaries.append(response)

    return text_summaries

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_img_summaries(imgPath, modelPath, prompt):
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

    model = MiniCPM_Llama3_int4(model_path=modelPath, max_memory_map={0: "10GB"})

    msgs = prompt

    # Apply to images
    for img_file in sorted(os.listdir(imgPath)):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(imgPath, img_file)
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

def runTextSummary(textPath, chunkSize, savePath):
    with open(textPath, "r") as f:
        texts = f.read()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunkSize, chunk_overlap=chunkSize*0.1
    )

    texts_token = text_splitter.split_text(texts)
    print("texts_token:", texts_token)

    text_summaries = generate_text_summaries(texts_token)

    tokenSavePath = os.path.join(savePath, 'text.pkl')
    summarySavePath = os.path.join(savePath, 'summary.pkl')

    with open(tokenSavePath, 'wb') as f:
        pickle.dump(texts_token, f)

    with open(summarySavePath, 'wb') as f:
        pickle.dump(text_summaries, f)

def runImgSummary(imgPath, modelPath, prompt, imgBase64SavePath, imgURLSumSavePath):
    img_base64_list, image_summaries_list, image_path_list = generate_img_summaries(imgPath,
                                                                                    modelPath,
                                                                                    prompt)

    # save img_base64_list
    save_to_pkl_path = imgBase64SavePath
    with open(save_to_pkl_path, 'wb') as f:
        pickle.dump(img_base64_list, f)

    # save image_path_list, image_summaries_list
    save_to_json_path = imgURLSumSavePath
    save_to_json(image_path_list, image_summaries_list, save_to_json_path)



if __name__ == "__main__":
    # Get text summaries
    # text_path = '../data/1500/doc/SIMATIC S7-1200-1500编程指南.txt'
    # chunk_size = 1000
    # save_path = '../data/1500/doc/1000token'
    #
    # runTextSummary(text_path, chunk_size, save_path)

    # ==============================
    # Get image summaries
    img_path = "../data/1500/figures"
    model_path = "../model_pool/MiniCPM-Llama3-V-2_5-int4"
    img_summary_prompt = """以西门子PLC为背景，详细总结图中的信息。"""
    img_base64_save_path = '../data/1500/img_base64_list.pkl'
    img_URL_sum_save_path = "../data/1500/path_summaries_list.json"

    runImgSummary(img_path,
                  model_path,
                  img_summary_prompt,
                  img_base64_save_path,
                  img_URL_sum_save_path)
