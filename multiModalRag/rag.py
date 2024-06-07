import io
import re

from langchain_core.output_parsers import StrOutputParser
import base64
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_community.document_transformers import LongContextReorder

from multiModalRag.embedding import build_multiModal_retriever
from model_pool.LLM import ChatGLM
# from embedding import build_multiModal_retriever
# from chatApp.model_pool.LLM import ChatGLM


def display_image_from_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    plt.imshow(image)
    plt.axis('off')
    plt.show()

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    most_related_type = None
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            # doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
            if most_related_type == None:
                most_related_type = "image"
        else:
            texts.append(doc)
            if most_related_type == None:
                most_related_type = "text"
    return {"images": b64_images, "texts": texts, "most_related_type": most_related_type}




if __name__ == "__main__":
    text_fpath = '../data/1200/doc/1000token'
    img_fpath = '../data/1200'
    embed_model_path = '../model_pool/bge-m3'

    retriever_multi_vector_img = build_multiModal_retriever(text_fpath, img_fpath, embed_model_path)

    # print("Start building LLMs...")
    model_dir = "../model_pool/chatglm3-6b"
    # max_memory_map = {0: "11GB", 1: "11GB"}
    # model = ChatGLM(model_dir, max_memory_map)
    # print("Finish model building!")
    # print("=" * 20)

    # =========RAG==========
    reording = LongContextReorder()
    # question = "s7-1200有哪些型号？"
    question = "s7-1200的完整扫描周期中执行的任务过程有哪些？"

    docs = retriever_multi_vector_img.invoke(question)
    # print("docs:", docs)
    # reorder_docs = reording.transform_documents(docs)
    # print("reording:", docs)
    img_text_dict = split_image_text_types(docs)
    print(img_text_dict)
    # image = img_text_dict['images'][0]

    # =========Conversation=========
    # print("***User***: \n %s" % question)
    # if len(img_text_dict['texts']) != 0:
    #     text = img_text_dict['texts'][0]
    #     response, history = model.chat(question, history=[], context=text)
    #     print("***AI(本地知识库检索)***: \n %s" % response)
    #     print("\n" * 1)
    # else:
    #     response, history = model.chat(question, history=[])
    #     print("***AI(无检索)***: \n %s" % response)
    #     print("\n" * 1)
    #
    # display_image_from_base64(image)
