import os
import glob

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from tqdm import tqdm

def load_pdf_file(file_path):
    # loader = UnstructuredPDFLoader(file_path, mode="elements")
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    return docs
    # [doc(page_content = 'str', meta_data = {'source': 'file path'})]


def creat_chunk_list(data_path, splitter):
    file_path_lists = glob.glob(os.path.join(data_path, "*.pdf"))
    chunk_list = []
    doc_list = []
    for file_path in tqdm(file_path_lists):
        print(f"Indexing {file_path}...")
        doc = load_pdf_file(file_path)
        doc_list.append(doc[0])
        chunk_list.extend(splitter.split_text(doc[0].page_content))

    return doc_list, chunk_list



if __name__ == "__main__":
    # file_path = "../data/siemens/S7-400产品手册.pdf"
    data_path = "../data"
    # docs = load_pdf_file(file_path)
    # print(docs)

    c_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )

    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    doc_list, chunk_list = creat_chunk_list(data_path, rc_splitter)
    print(doc_list)
    print("=" * 50)
    print(chunk_list)
    for chunk in chunk_list[:10]:
        print(chunk)