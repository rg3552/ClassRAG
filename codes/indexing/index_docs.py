import argparse
import os
import glob
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    # Uncomment when available
    # UnstructuredPPTXLoader,
    # UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from codes.utils import load_embeddings, get_vector_store


def load_txt_files(file_path, text_splitter):
    # loader = TextLoader(file_path)
    loader = UnstructuredLoader(file_path)
    documents = loader.load()
    return text_splitter.split_documents(documents)


def load_pdf_files(file_path, text_splitter):
    # loader = PyPDFLoader(file_path)
    loader = UnstructuredLoader(file_path)
    documents = loader.load()
    return text_splitter.split_documents(documents)


def index_documents(root_dir, extensions, persist_directory):
    """
    Index documents in the specified directory with given extensions and save the index locally.

    Parameters:
        root_dir (str): The root directory containing documents.
        extensions (list): List of file extensions to include (e.g., ['txt', 'pdf', 'pptx', 'xlsx']).
        persist_directory (str): Directory where the Chroma index will be saved.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = load_embeddings(model_name)

    # Initialize Chroma vector store with persistence enabled
    vector_store = get_vector_store(persist_directory, "class_info", embeddings)

    # Define chunk size and overlap
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Mapping extensions to their respective loading functions
    loader_mapping = {
        "txt": load_txt_files,
        "pdf": load_pdf_files,
        # "pptx": load_pptx_files,
        # "xlsx": load_xlsx_files,
    }

    # Iterate over each file extension and load files
    for ext in extensions:
        ext = ext.lstrip(".")  # Remove the leading dot from extensions for consistency
        if ext not in loader_mapping:
            print(f"Unsupported file extension: {ext}")
            continue

        # Get all files with the current extension
        file_paths = glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True)

        print(f"Processing files: {file_paths}")
        # Load documents, chunk them, and add to the vector store
        loader_function = loader_mapping[ext]
        documents = loader_function(file_paths, text_splitter)
        documents = filter_complex_metadata(documents)
        print(documents[0], len(documents))
        vector_store.add_documents(documents)

    print(f"Indexing completed and saved to {persist_directory}.")


def main():
    parser = argparse.ArgumentParser(description="Index documents in a directory.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing documents.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        required=True,
        help="List of file extensions to include (e.g., txt pdf).",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_index",
        help="Directory to save the Chroma index.",
    )

    args = parser.parse_args()

    # print(args)

    index_documents(args.root_dir, args.extensions, args.persist_directory)


if __name__ == "__main__":
    main()
