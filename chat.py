from txtai.pipeline import LLM, Textractor, Summary
from txtai import Embeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import os
import argparse

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_TEMPLATE = dict(
    RAG_CHATGLM_TEMPLATE="""
  <|system|>
  You are a friendly assistant. You answer questions from users.
  Answer the following question according to the context below. Only include information specifically discussed
  The answer should not includes thinking processes and other irrelated information.
  <|user|>
  question: {question}
  context: {text}
  <|assistant|>
  """,

    CHATGLM_TEMPLATE="""
  <|system|>
  You are a friendly assistant. You answer questions from users.
  Answer the following question. Only include information specifically discussed.
  The answer should not includes thinking processes and other irrelated information.
  <|user|>
  question: {question}
  <|assistant|>
  """,

    AUTO_RAG_CHATGLM_TEMPLATE="""
  <|system|>
  You are a friendly assistant. You answer questions from users.
  Answer the following question according to the context below. Only include information specifically discussed.
  If there is no relevant information in the provided context, try to answer yourself.
  The answer should not includes thinking processes and other irrelated information.
  <|user|>
  question: {question}
  context: {text}
  <|assistant|>
  """,

)


def stream(path):
    for f in sorted(os.listdir(path)):
        fpath = os.path.join(path, f)

        # Only accept documents
        if f.endswith(("docx", "xlsx", "pdf")):
            print(f"Indexing {fpath}")
            for chunk in textractor(fpath):
                yield chunk


def context(question, embeddings):
    context = "\n".join(x["text"] for x in embeddings.search(question, limit=2))
    return context


def loadIndex(database_path, index_path, embeddings, is_update=False):
    if is_update:
        print("Create new vector indexing...")
        if not os.path.exists(index_path): os.mkdir(index_path)
        embeddings.index(stream(database_path))
        embeddings.save(index_path)
    else:
        print("Load vector indexing...")
        embeddings.load(index_path)


if __name__ == "__main__":

    # =========GPU setting=========
    max_memory_map = {0:"11GB", 1:"11GB"}

    # =========Local database path=========
    local_database_path = "./data/siemens"
    index_save_path = "data/indexing_plc"

    # Document text extraction, split into chunks (e.g. sentences/paragraphs). Vector database.
    print("Extracting documents...")
    textractor = Textractor(paragraphs=True)

    embeddings = Embeddings(content=True)
    loadIndex(local_database_path, index_save_path, embeddings)

    # =========Create model=========
    print("Start building LLMs...")
    model_dir = "./model_pool/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, max_memory=max_memory_map, device_map='auto')
    model = model.eval()
    print("Finish model building!")
    print("=" * 20)

    # =========Conversation=========
    question = "simatic energy manager v7.5上市时间是多少？"
    text = context(question, embeddings)
    prompt = PROMPT_TEMPLATE["RAG_CHATGLM_TEMPLATE"].format(question=question, text=text)
    response, history = model.chat(tokenizer, question, question, history=[{"role": "system", "content": prompt}])
    print("***User***: \n %s" % question)
    print("***AI(本地知识库检索)***: \n %s" % response)
    print("\n" * 1)

    # question = "将上面的回答翻译成英文。"
    # prompt = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question=question)
    # response, history = model.chat(tokenizer, question, question, history=history)
    # print("***User***: \n %s" % question)
    # print("***AI(无本地知识库)***: \n %s" % response)
    # print("\n" * 1)
    #
    # print(history)

    #
    # question = "回答的不够好，再用100字的中文详细回答一次。"
    # # text = context(question, embeddings)
    # prompt = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question = question)
    # response, history = model.chat(tokenizer, prompt, history=history)
    # print("***User***: \n %s" % question)
    # print("***AI***: \n %s" % response)
    # print("\n" * 1)
    #
    #
    question = "从设备组态中怎样查找MAC地址？"
    prompt = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question = question)
    response, _ = model.chat(tokenizer, prompt, prompt, history=[])
    print("***User***: \n %s" % question)
    print("***AI***: \n %s" % response)

    print("\n" * 1)

    text = context(response, embeddings)
    prompt = PROMPT_TEMPLATE["RAG_CHATGLM_TEMPLATE"].format(question=question, text= text)
    # prompt = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question=question + response + text)
    response, _ = model.chat(tokenizer, prompt, prompt, history=[])
    print("***User***: \n %s" % question)
    print("***AI***: \n %s" % response)

    print("\n" * 1)
    print("retrivel results:", text)