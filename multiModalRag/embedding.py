import os.path
import pickle
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
import json
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder

import base64
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO


def load_text(fpath):
    text_path = os.path.join(fpath, 'text.pkl')
    text_summary_path = os.path.join(fpath, 'summary.pkl')

    with open(text_path, 'rb') as f_text:
        texts = pickle.load(f_text)
    with open(text_summary_path, 'rb') as f_summary:
        text_summaries = pickle.load(f_summary)
    return texts, text_summaries

def load_imgs(fpath):
    img_path = os.path.join(fpath, 'path_summaries_list.json')
    img_base64_path = os.path.join(fpath, 'img_base64_list.pkl')

    # load data from json
    with open(img_path, 'r') as file:
        data = json.load(file)
    # split data
    image_path_list = list(data.keys())
    image_summaries = list(data.values())

    with open(img_base64_path, 'rb') as img_base64:
        img_base_64_list = pickle.load(img_base64)

    return img_base_64_list, image_path_list, image_summaries

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents_text(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, summary_docs)))

    def add_documents_table_or_img(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents_text(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents_table_or_img(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents_table_or_img(retriever, image_summaries, images)

    return retriever

def build_multiModal_retriever(text_fpath, img_fpath, embed_model_path):
    texts, text_summaries = load_text(text_fpath)
    img_base64_list, _, image_summaries = load_imgs(img_fpath)
    tables, table_summaries = [], []

    embed_model = HuggingFaceEmbeddings(model_name=embed_model_path)

    vectorstore = Chroma(
        collection_name="mm_rag_s7-1200", embedding_function=embed_model
    )

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    return retriever_multi_vector_img

def build_parentChunk_retriever(full_text_fpath, embed_model_path, parent_chunkSize, child_chunkSize):
    parent_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=parent_chunkSize)
    child_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=child_chunkSize)

    embed_model = HuggingFaceEmbeddings(model_name=embed_model_path)
    vectorstore = Chroma(
        collection_name="mm_rag_s7-1200", embedding_function=embed_model
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    # Add docs
    docs = []
    docs.extend(TextLoader(full_text_fpath).load())

    retriever.add_documents(docs, ids=None)
    return retriever

if __name__ == "__main__":

    text_fpath = '../data/1200/doc/500token'
    img_fpath = '../data/1200'
    full_text_fpath = '../data/1200/doc/s7-1200文档.txt'
    embed_model_path = '../model_pool/bge-m3'

    # question = "如何在step 7 basic中创建一个项目?"
    question = "s7-1200是什么？"

    # retriever_multiModal = build_multiModal_retriever(text_fpath,
    #                                                   img_fpath,
    #                                                   embed_model_path)

    retriever_parentChunk = build_parentChunk_retriever(full_text_fpath,
                                                        embed_model_path,
                                                        parent_chunkSize=500,
                                                        child_chunkSize=200
                                                        )

    res_parentChunk = retriever_parentChunk.invoke(question)

    # res_multiModal = retriever_multiModal.invoke(question)
    # res_with_score = retriever_multiModal.vectorstore.similarity_search_with_score(question)
    # reorder_res_with_score = LongContextReorder().transform_documents(res_with_score)
    #
    # print(res_multiModal[0])
    # print('==========')
    # print(res_with_score[0][0].page_content)
    # print('==========')
    # print(res_with_score[0][1])
    # print('===========')
    # print(reorder_res_with_score)



    # 兩個檢索的結果一致，可返回score。

