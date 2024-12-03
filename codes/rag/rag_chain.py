from dotenv import load_dotenv

# from langchain import hub
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from operator import itemgetter
from ..utils import load_embeddings, get_vector_store
import math
from datetime import datetime

RAG_PROMPT_TEMPLATE = """
You are a course assistant for a graduate-level course at Columbia University titled “Practical Deep Learning System Performance”. Your role is to assist students by answering their queries related to logistics and academic doubts.

The course details, policies, and content may change over time, so it is crucial to refer to the most recent and most relevant documents while forming your response. Below, you will be provided context chunks from various document types, such as:
- Lecture transcripts
- Lecture slides
- Syllabus
- Assignments
- Other course-related material

Each context chunk includes a type (e.g., "lecture transcript") and the date it was last modified. Use these details to prioritize the most up-to-date and contextually relevant information when answering the query.

### Instructions:
1. Read the query carefully to understand the student's request.  
2. Select the most relevant context chunks from the provided materials, prioritizing the latest updates where applicable.  
3. Formulate a concise and accurate response based on the chosen context, making it clear and helpful for the student.

Here’s the student’s query:  
{question}

Relevant context chunks:  
{context}

Your response should use the above context to provide the most accurate and up-to-date answer. If the context lacks sufficient information, respond by stating that you cannot find the exact answer but guide the student on how to get it.
"""


def initialize_retriever(persist_directory, collection_name, embeddings):
    """
    Initialize a Chroma retriever from a specified directory and embeddings model.
    """
    vector_store = get_vector_store(persist_directory, collection_name, embeddings)
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 7, "fetch_k": 15}
    )
    return retriever


def exponential_decay(modified_date, decay_rate=0.01):
    days_old = (
        datetime.now() - datetime.fromisoformat(modified_date)
    ).total_seconds() / (24 * 3600)
    # print(days_old)
    return math.exp(-decay_rate * days_old)


def re_rank_with_decay(retrieved_docs):
    # Create a list of tuples with document and adjusted score
    ranked_docs = []
    for i, doc in enumerate(retrieved_docs):
        decay_factor = exponential_decay(doc.metadata.get("last_modified"))
        # Use the position in the current list as a proxy for similarity score (higher = better similarity)
        # initial_score = len(retrieved_docs) - i  # Higher similarity closer to the start
        # final_score = initial_score * decay_factor
        final_score = decay_factor
        ranked_docs.append((doc, final_score))

    # Sort by the computed final score
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs]


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        # Extract metadata for the document
        metadata = doc.metadata

        # Construct the context with metadata
        formatted_doc = (
            f"Category: {metadata.get('category', 'N/A')}\n"
            # f"Element ID: {metadata.get('element_id', 'N/A')}\n"
            # f"Filename: {metadata.get('filename', 'N/A')}\n"
            f"Last Modified: {metadata.get('last_modified', 'N/A')}\n"
            f"Source: {metadata.get('source', 'N/A')}\n"
            f"Content: {doc.page_content}\n"
        )

        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def setup_rag_chain(
    persist_directory="./chroma_index",
    collection_name="class_info",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    cohere_model="command-r-plus-08-2024",
    openai_model="gpt-4o-mini",
):
    load_dotenv()
    embeddings = load_embeddings(embedding_model)

    retriever = initialize_retriever(persist_directory, collection_name, embeddings)

    # Create a PromptTemplate object
    rag_prompt = PromptTemplate(
        input_variables=[
            "question",
            "context",
        ],  # Variables that will be substituted in the template
        template=RAG_PROMPT_TEMPLATE,
    )
    llm = ChatCohere(model=cohere_model, temperature=0)
    # llm = ChatOpenAI(
    #     model=openai_model, temperature=0, max_tokens=None, timeout=None, max_retries=2
    # )

    rag_chain = (
        RunnablePassthrough()
        | {
            "context": retriever | re_rank_with_decay | format_docs,
            "question": RunnablePassthrough(),
        }
        | {
            "output": RunnablePassthrough() | rag_prompt | llm | StrOutputParser(),
            "question": itemgetter("question"),
            "context": itemgetter("context"),
        }
    )

    return rag_chain
