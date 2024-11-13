from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_embeddings(model_name, device="cpu"):
    """
    Load HuggingFace embeddings with specified model and device.
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def initialize_retriever(persist_directory, collection_name, embeddings):
    """
    Initialize a Chroma retriever from a specified directory and embeddings model.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 7, "fetch_k": 15}
    )
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def setup_rag_chain(
    persist_directory,
    collection_name,
    embedding_model,
    device="cpu",
    cohere_model="command-r-plus-08-2024",
    openai_model="gpt-4o-mini",
):
    load_dotenv()
    embeddings = load_embeddings(embedding_model, device)
    retriever = initialize_retriever(persist_directory, collection_name, embeddings)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatCohere(model=cohere_model, temperature=0)
    # llm = ChatOpenAI(
    #     model=openai_model, temperature=0, max_tokens=None, timeout=None, max_retries=2
    # )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
