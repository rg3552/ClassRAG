import random
import re
import asyncio
import nest_asyncio
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from ..utils import get_vector_store, load_embeddings
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI

# Apply nested asyncio compatibility
# nest_asyncio.apply()

# Define the question generation template
QUESTION_GENERATION_TEMPLATE = """
You are tasked with generating test questions to evaluate the performance of a Retrieval-Augmented Generation (RAG) module designed as a course assistant for a graduate-level course titled “Practical Deep Learning System Performance” at Columbia University.

The RAG module is expected to answer both academic and logistical queries based on course materials such as:
- Lecture transcripts
- Lecture slides
- Syllabus
- Assignments
- Course policies
- Administrative updates
---------------------
{context}
---------------------
### Instructions:

1. **Question Guidelines**:
   - The questions should require the module to identify and use specific, relevant details from the provided context.
   - Questions may involve:
     - Clarifications about course policies (e.g., late submission rules, grading criteria).
     - Academic queries related to technical content (e.g., concepts taught in lectures, assignments).
     - Hypothetical scenarios that test reasoning or edge cases (e.g., “What if a student needs an extension for an assignment?”).
     - Questions requiring cross-referencing multiple pieces of context (e.g., linking an assignment requirement to lecture material).
   - Ensure the questions are challenging enough to thoroughly test the module’s retrieval and response accuracy.

2. **Answer Guidelines**:
   - Provide an ideal answer to each question, based only on the context described above.
   - The answer should include reasoning for correctness, pointing to specific details or context pieces where applicable.

### Format:
Generate {num_questions} questions and a correct answer only using 'Answer' and 'Question' keys as per the format below:

[Question 1]
Question: 
Answer:

[Question 2]
Question:
Answer:

Do not include any additional keys or information in the response. 
"""


def generate_questions(llm, section, num_questions=1):

    question_generation_prompt = ChatPromptTemplate.from_template(
        QUESTION_GENERATION_TEMPLATE, partial_variables={"num_questions": num_questions}
    )

    question_generation_chain = question_generation_prompt | llm

    response = question_generation_chain.invoke({"context": section["documents"][0]})

    question_re = re.compile(
        r"\[Question \d+\]\s+Question:\s*(.*?)\s+Answer:\s*(.*?)(?=\s*\[Question \d+\]|\Z)"
    )
    q_n_a = question_re.findall(response.content)
    result = []
    for q, a in q_n_a:
        result.append(
            {
                "ref_doc_id": section["metadatas"][0]["element_id"],
                "question": q,
                "answer": a,
                "context": section["documents"],
            }
        )

    return result


def run_question_generation(llm, vector_store, num_questions=50):
    doc_ids = vector_store.get()["ids"]
    selected_doc_ids = random.sample(doc_ids, num_questions)
    results = []
    for doc_id in selected_doc_ids:
        if vector_store.get(doc_id)["metadatas"][0]["category"] == "NarrativeText":
            # print("yes")
            results.append(
                generate_questions(
                    llm,
                    vector_store.get(doc_id),
                    num_questions=2,
                )
            )
    # print(results)
    return results


# async def generate_questions(llm, section, num_questions=1):
#     """
#     Asynchronous function to generate questions using the given context.
#     """

#     question_generation_prompt = ChatPromptTemplate.from_template(
#         QUESTION_GENERATION_TEMPLATE, partial_variables={"num_questions": num_questions}
#     )
#     question_generation_chain = question_generation_prompt | llm

#     response = await question_generation_chain.ainvoke(
#         {"context": section["documents"][0]}
#     )
#     question_re = re.compile(r"\[Question \d+\]\nQuestion: (.*)\nAnswer: (.*)\n?")
#     q_n_a = question_re.findall(response.content)

#     result = []
#     for q, a in q_n_a:
#         result.append(
#             {
#                 "ref_doc_id": section["metadatas"][0]["element_id"],
#                 "question": q.strip(),
#                 "answer": a.strip(),
#                 "context": section["documents"],
#             }
#         )
#     return result


# def run_question_generation(llm, vector_store, num_questions=100):
#     """
#     Synchronous function to generate questions for selected documents.
#     """
#     doc_ids = vector_store.get()["ids"]
#     selected_doc_ids = random.sample(doc_ids, num_questions)

#     tasks = [
#         generate_questions(
#             llm,
#             vector_store.get(doc_id),
#             num_questions=2,
#         )
#         for doc_id in selected_doc_ids
#         if vector_store.get(doc_id)["metadatas"][0]["category"] == "NarrativeText"
#     ]
#     event_loop = asyncio.get_event_loop()
#     results = event_loop.run_until_complete(asyncio.gather(*tasks))

#     return results


import argparse


def main():
    """
    Main function to execute the question generation and save results.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Generate questions for RAG module evaluation."
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=100,
        help="Number of question/answer pairs to generate",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_index",
        help="Path to the directory where vector store is persisted.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="class_info",
        help="Name of the collection to use in the vector store.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_data/generated_QnA.csv",
        help="Path to save the generated Q&A CSV file.",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load embeddings and vector store
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = load_embeddings(model_name)
    vector_store = get_vector_store(
        args.persist_directory, args.collection_name, embeddings
    )

    # cohere_model = "command-r-plus-08-2024"
    # llm = ChatCohere(model=cohere_model, temperature=0)
    llm = ChatOpenAI(
        model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2
    )

    # Generate questions
    QA_pairs = run_question_generation(llm, vector_store, args.num_questions)
    QA_pairs = [item for sublist in QA_pairs for item in sublist]
    QA_pairs_df = pd.DataFrame(QA_pairs)

    # Save to the specified output path
    QA_pairs_df.to_csv(args.output_path, index=False)
    print(f"Generated Q&A saved to {args.output_path}")


if __name__ == "__main__":
    main()
