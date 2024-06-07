from txtai.pipeline import Textractor
from txtai import Embeddings
import os
import glob
from parser_file import creat_chunk_list
# from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VectorStore:
    def __init__(self, database_path, index_path, textractor, embeddings, splitter):
        self.database_path = database_path
        self.index_path = index_path
        self.textractor = textractor
        self.embeddings = embeddings
        self.splitter = splitter

    def stream(self, database_path) -> None:
        for f in sorted(os.listdir(database_path)):
            fpath = os.path.join(database_path, f)

            # Only accept documents
            if f.endswith(("docx", "xlsx", "pdf")):
                print(f"Indexing {fpath}")
                for chunk in self.textractor(fpath):
                    # print(chunk)
                    yield chunk

    def custom_stream(self, database_path, splitter):
            doc_list, chunk_list = creat_chunk_list(database_path, splitter)
            for chunk in chunk_list:
                yield chunk

    def loadIndex(self, create_new) -> None:
        if create_new:
            print("Create new vector indexing...")
            if not os.path.exists(self.index_path): os.mkdir(self.index_path)
            # self.embeddings.index(self.stream(self.database_path))
            self.embeddings.index(self.custom_stream(self.database_path, self.splitter))
            self.embeddings.save(self.index_path)
        else:
            print("Load vector indexing...")
            self.embeddings.load(self.index_path)

    def build(self, create_new=True):
        self.loadIndex(create_new)



if __name__ == "__main__":
    local_database_path = "/home/ubuntu/Desktop/code/chatApp/data/siemens"
    index_save_path = "../data/indexing_plc"

    model_dir = "../model_pool/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, max_memory={0:"11GB", 1:"11GB"},
                                                 device_map='auto')
    # model = model.quantize(4)
    model = model.eval()

    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=2000,
        chunk_overlap=50,
        length_function=len,
    )
    textractor = Textractor(paragraphs=True)
    embeddings = Embeddings(content=True)

    vectorStore = VectorStore(local_database_path, index_save_path, textractor, embeddings, rc_splitter)
    vectorStore.build()

    # ============= query eval
    embeddings.load(index_save_path)
    # results = embeddings.search("simatic energy manager v7.5上市时间是多少？", limit=3)
    # query = "s7-200有什么新功能?"
    # results = model.chat()
    # results = embeddings.search( + query, limit=3)
    # for x in results:
    #     print("{} \n".format(x))

    # path_list = glob.glob(os.path.join(local_database_path, "*.docx"))
    # for path in path_list[:1]:
    #     print(path)
    #     file_name = os.path.split(path)[-1].split(".")[0]
    #     with open(path.replace("docx", "txt"), "w") as f:
    #         for chunk in textractor(path):
    #             chunk = chunk.replace('\n', '').replace('\r', '')
    #             print(chunk)
    #             f.write( "=====" + chunk + '\n\n')

