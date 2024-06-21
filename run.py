import torch
from multiModalRag.embedding import build_multiModal_retriever, build_parentChunk_retriever
from multiModalRag.rag import split_image_text_types
import matplotlib.pyplot as plt
import numpy as np
from model_pool.langchainLLM import ChatGLM4_LLM
from model_pool.LLM import MiniCPM_Llama3_int4, ChatGLM, MiniCPM_Llama3
import base64
from PIL import Image
from io import BytesIO
from langchain_community.document_transformers import LongContextReorder
import json

def get_glm4(model_path, gpu_device, gen_kwargs):
    model = ChatGLM4_LLM(model_path, gpu_device, gen_kwargs)
    return model

def get_cpm(model_path, max_memory_map):
    model = MiniCPM_Llama3(model_path, max_memory_map)
    return model

def get_multiModal_retriever(text_fpath, img_fpath, embed_model_path):
    retriever = build_multiModal_retriever(text_fpath, img_fpath, embed_model_path)
    return retriever

def get_parentChunk_retriever(full_text_fpath,
                              embed_model_path,
                              parent_chunkSize=500,
                              child_chunkSize=200):

    retriever = build_parentChunk_retriever(full_text_fpath,
                                            embed_model_path,
                                            parent_chunkSize,
                                            child_chunkSize)
    return retriever

def convert_image_from_base64(base64_str):
    # base64 to bytes
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def loadModel(args):
    cpm = get_cpm(
        model_path=args["cpm_path"],
        max_memory_map=args["cpm_max_memory_map"]
    )

    glm4 = get_glm4(
        model_path=args["glm4_path"],
        gpu_device=args["glm4_gpu_device"],
        gen_kwargs=args["glm4gen_kwargs"]
    )

    return cpm, glm4

def buildRetriever(args):
    retriever_multiModal=get_multiModal_retriever(
        text_fpath=args["text_fpath"],
        img_fpath=args["img_fpath"],
        embed_model_path=args["embed_model_path"]
    )
    retriever_parentChunk=get_parentChunk_retriever(full_text_fpath=args["full_text_fpath"],
                                                    embed_model_path=args["embed_model_path"]
                                                    )

    return retriever_multiModal, retriever_parentChunk

def retrieve(question, retriever_multiModal, retriever_parentChunk, multiModal_model):

    # multimodal rag
    res_multiModal_with_score = retriever_multiModal.vectorstore.similarity_search_with_score(question)
    res_multiModal_score = res_multiModal_with_score[0][1]
    res_multiModal_rag_text = ""

    if res_multiModal_score < 0.6:
        res_multiModal = retriever_multiModal.invoke(question)
        res_multiModal_reorder = LongContextReorder().transform_documents(res_multiModal)

        res_multiModal_img_text_dict = split_image_text_types(res_multiModal_reorder)
        res_multiModal_most_related_type = res_multiModal_img_text_dict["most_related_type"]
        print("====multiModal rag doc====:", res_multiModal_img_text_dict['texts'])

        if res_multiModal_most_related_type == "image":
            # cpm
            res_multiModal_img_base64 = res_multiModal_img_text_dict['images'][0]
            res_multiModal_rag_image = convert_image_from_base64(res_multiModal_img_base64)
            res_multiModal_rag_text = multiModal_model.chat(res_multiModal_rag_image, question)
            print("====RAG texts from image====:", res_multiModal_rag_text)

        elif res_multiModal_most_related_type == "text":
            res_multiModal_rag_text = res_multiModal_img_text_dict['texts'][0]
            print("====RAG texts from documents====:", res_multiModal_rag_text)

    # parent chunk rag
    res_parentChunk = retriever_parentChunk.invoke(question)
    if len(res_parentChunk) == 0:
        res_parentChunk = ""
    else:
        res_parentChunk = res_parentChunk[0].page_content
    print(f"====RAG text parent chunk====: {res_parentChunk} \n")

    final_rag_info = str({"上下文信息": res_parentChunk + '\n' + res_multiModal_rag_text})
    print("====RAG final resutls====:", final_rag_info)
    return final_rag_info, res_multiModal_most_related_type, res_multiModal_rag_image

def build_chat_history(history, question, answer, multiModal_res_type, res_multiModal_rag_image, rag_img_path = None):
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "rag_img_path": rag_img_path}
         ]
    )

    if multiModal_res_type == "image":
        history[-1]['rag_img_path'] = res_multiModal_rag_image
    else:
        history[-1]['rag_img_path'] = None

    return history

def run(args,question):

    retriever_multiModal, retriever_parentChunk = buildRetriever(args)
    cpm, glm4 = loadModel(args)

    rag_info, multiModal_res_type, res_multiModal_rag_image = retrieve(args, retriever_multiModal, retriever_parentChunk, cpm)

    history = []
    response = glm4.invoke(
        question=question,
        chat_history=history,
        context=rag_info,
        prompt_template=args["prompt_template"]
    )

    history=build_chat_history(history,question,response, multiModal_res_type, res_multiModal_rag_image)

    return response, history

if __name__ == "__main__":
    config_path = "./config.json"
    args = json.load(config_path)

    question = "who?"
    response, history = run(args, question)


