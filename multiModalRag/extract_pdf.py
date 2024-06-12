from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
Extract PDF, save figures and txt.
"""
# Extract elements from PDF

def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        url = None,
        strategy= "hi_res",
        filename=path + fname,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=output_image_save_path,
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


if __name__ == "__main__":
    # File path
    fpath = "../data/1500/"
    fname = "SIMATIC S7-1200-1500编程指南.pdf"

    output_image_save_path = fpath + 'figures'

    # Get elements
    raw_pdf_elements = extract_pdf_elements(fpath, fname)
    print("raw_pdf_elements:", raw_pdf_elements)
    print("=" * 40)

    # Get text, tables
    texts, tables, images = categorize_pdf_elements(raw_pdf_elements)
    print("texts:", texts)
    print("=" * 40)
    print("tables:", tables)
    print("=" * 40)
    print("images:", images)
    print("=" * 40)

    with open(fpath + fname.replace("pdf", "txt"), "w") as f:
        for text in texts:
            f.write(text)

