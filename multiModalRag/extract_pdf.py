from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
Extract PDF, save figures and txt.
"""
# Extract elements from PDF

def extract_pdf_elements(file_path, img_save_path):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        url = None,
        strategy= "hi_res",
        filename=file_path,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=img_save_path,
    )


# Categorize elements by type
def categorize_pdf_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    images = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
        elif "unstructured.documents.elements.Image" in str(type(element)):
            images.append(str(element))

    return texts, tables, images

def runPDFExtract(file_path, text_save_path, img_save_dir):

    raw_pdf_elements = extract_pdf_elements(file_path, img_save_dir)
    print("raw_pdf_elements:", raw_pdf_elements)
    print("=" * 40)

    texts, tables, images = categorize_pdf_elements(raw_pdf_elements)

    with open(text_save_path, "w") as f:
        for text in texts:
            f.write(text)

    return texts, tables, images


if __name__ == "__main__":
    # File path
    file_path = "../data/source/siemens/SIMATIC Energy Manager V7.5上市通知.pdf"
    text_save_path = "../data/rag/fullText/SIMATIC Energy Manager V7.5上市通知.txt"
    img_save_dir = "../data/rag/extract_img/img/SIMATIC Energy Manager上市通知"

    texts, tables, images = runPDFExtract(file_path, text_save_path, img_save_dir)
    print("texts:", texts)
    print("=" * 40)
    print("tables:", tables)
    print("=" * 40)
    print("images:", images)
    print("=" * 40)
