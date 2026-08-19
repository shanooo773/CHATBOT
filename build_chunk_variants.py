"""
Build alternate FAISS indices over the same source PDF with different
chunking strategies, so rag_eval.py can compare them head-to-head.

Variants:
  baseline  - 500 chars / 50 overlap  (what memory.py already builds)
  wide      - 1000 chars / 200 overlap, same RecursiveCharacterTextSplitter
  sentence  - NLTK sentence-aware splitter, ~1000 char target, never
              splits a sentence across a chunk boundary

Run: python build_chunk_variants.py
"""

import nltk

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def build(name, splitter, documents, embedding_model):
    print(f"\n=== building '{name}' ===")
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks")
    db = FAISS.from_documents(chunks, embedding_model)
    out_path = f"vectorstore/db_faiss_{name}"
    db.save_local(out_path)
    print(f"  saved to {out_path}")
    return len(chunks)


def main():
    print("Loading PDF(s) from data/...")
    documents = load_pdf_files(DATA_PATH)
    print(f"  {len(documents)} pages")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    counts = {}
    counts["wide_1000_200"] = build(
        "wide_1000_200",
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        documents,
        embedding_model,
    )
    counts["sentence_1000"] = build(
        "sentence_1000",
        NLTKTextSplitter(chunk_size=1000),
        documents,
        embedding_model,
    )

    print("\nChunk counts:", counts)
    print("(baseline 500/50 chunk count is whatever vectorstore/db_faiss was built with)")


if __name__ == "__main__":
    main()
