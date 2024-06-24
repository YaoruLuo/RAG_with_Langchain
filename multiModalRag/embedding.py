import glob
import os.path
import pickle
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder


def load_text(token_fpath):
    texts = []
    text_summaries = []

    text_path_list = glob.glob(os.path.join(token_fpath, 'chunck', '*.pkl'))

    for text_path in text_path_list:
        with open(text_path, 'rb') as f_text:
            text = pickle.load(f_text)
            texts += text

        with open(text_path.replace('chunck', 'summary'), 'rb') as f_summary:
            text_summary = pickle.load(f_summary)
            text_summaries += text_summary
    return texts, text_summaries

def load_imgs(imgText_fpath):
    image_path_list = []
    image_summary_list = []
    img_base_64_list = []

    img_summary_path_list = glob.glob(os.path.join(imgText_fpath, 'path_summary', '*.json'))
    # load data
    for img_summary_path in img_summary_path_list:
        with open(img_summary_path, 'r') as file:
            data = json.load(file)
        # split data
        image_path_list += list(data.keys())
        image_summary_list += list(data.values())

        img_base64_path = img_summary_path.replace('path_summary', 'img_base64')
        img_base64_path = img_base64_path.replace('json', 'pkl')
        with open(img_base64_path, 'rb') as img_base64:
            img_base_64 = pickle.load(img_base64)
            img_base_64_list += img_base_64

    return img_base_64_list, image_path_list, image_summary_list

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, image_summaries, images
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

    def add_documents_img(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents_text(retriever, text_summaries, texts)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents_img(retriever, image_summaries, images)

    return retriever

def build_multiModal_retriever(dataBase_url, embed_model_path, token_size="500token"):
    token_fpath = os.path.join(dataBase_url, 'token', token_size)
    texts, text_summaries = load_text(token_fpath)

    imgText_fpath = os.path.join(dataBase_url, 'imgText')
    img_base64_list, _, image_summaries = load_imgs(imgText_fpath)

    embed_model = HuggingFaceEmbeddings(model_name=embed_model_path)

    vectorstore = Chroma(
        collection_name="ragDataBase", embedding_function=embed_model
    )

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        image_summaries,
        img_base64_list,
    )

    return retriever_multi_vector_img

def build_parentChunk_retriever(dataBase_url, embed_model_path, parent_chunkSize, child_chunkSize):
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
    loader = DirectoryLoader(dataBase_url, glob="./fullText/*.txt")
    docs = loader.load()

    retriever.add_documents(docs, ids=None)
    return retriever

if __name__ == "__main__":

    dataBase_url = '../data/rag'
    embed_model_path = '../model_pool/bge-m3'
    token_size = "500token"

    test_multiModal_retriever = True
    test_parentChunk_retriever = False

    # question = "如何在step 7 basic中创建一个项目?"
    question = "s7-1500有什么功能？"

    if test_multiModal_retriever:
        retriever_multiModal = build_multiModal_retriever(dataBase_url, embed_model_path, token_size)

        res_multiModal = retriever_multiModal.invoke(question)
        res_with_score = retriever_multiModal.vectorstore.similarity_search_with_score(question)
        reorder_res_with_score = LongContextReorder().transform_documents(res_with_score)
        print(res_with_score)
        print(reorder_res_with_score)

    if test_parentChunk_retriever:
        retriever_parentChunk = build_parentChunk_retriever(dataBase_url,
                                                            embed_model_path,
                                                            parent_chunkSize=500,
                                                            child_chunkSize=200
                                                            )

        res_parentChunk = retriever_parentChunk.invoke(question)
        print(res_parentChunk)




