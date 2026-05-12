import os
import fitz  # PyMuPDF


def load_pdf(file_path):

    doc = fitz.open(file_path)

    pages = []

    for i, page in enumerate(doc):

        text = page.get_text()

        if text.strip() == "":
            continue

        pages.append({
            "text": text,
            "page": i + 1,
            "source": os.path.basename(file_path)
        })

    return pages


def load_all_pdfs(pdf_dir="data/pdf"):

    all_pages = []

    for file in os.listdir(pdf_dir):

        if file.lower().endswith(".pdf"):

            path = os.path.join(pdf_dir, file)

            pages = load_pdf(path)

            all_pages.extend(pages)

    return all_pages
