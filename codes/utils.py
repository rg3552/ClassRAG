from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from operator import itemgetter

import math
from datetime import datetime


def load_embeddings(model_name, model_kwargs=None, encode_kwargs=None):
    """
    Load HuggingFace embeddings with specified model and device.
    """
    load_dotenv()
    if model_kwargs is None:
        model_kwargs = {"device": "cpu"}
    if encode_kwargs is None:
        encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def get_vector_store(persist_directory, collection_name, embeddings):
    """
    Initialize a Chroma retriever from a specified directory and embeddings model.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store
