from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def semantic_chunk(pages, show_progress=True):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ". ", ";"]
    )
    chunks = []
    page_iter = pages
    if show_progress:
        page_iter = tqdm(pages, desc="语义切分", unit="page")
    for page in page_iter:
        pieces = splitter.split_text(page["text"])
        for piece in pieces:
            chunks.append({
                "text": piece,
                "page": page["page"],
                "source": page["source"]
            })
    return chunks
