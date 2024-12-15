import pandas as pd
import argparse
import logging
from ..rag.rag_chain import setup_rag_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/rag_evaluation.log"),
        logging.StreamHandler(),
    ],
)


def get_rag_answers(chain, query):
    try:
        output = chain.invoke(query)
        return (output["output"], output["context"])
    except Exception as e:
        logging.error(f"Failed to process query: {query}. Error: {e}")
        return ("ERROR: Unable to process", "")


def main(input_file, index_path, output_file):
    try:
        # Load evaluation data
        logging.info(f"Loading evaluation data from {input_file}...")
        eval_data = pd.read_csv(input_file)
        logging.info("Evaluation data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load input file {input_file}. Error: {e}")
        return

    try:
        # Initialize the RAG chain
        logging.info(f"Setting up RAG chain with index path {index_path}...")
        chain = setup_rag_chain(index_path)
        logging.info("RAG chain setup successfully.")
    except Exception as e:
        logging.error(f"Failed to set up RAG chain. Error: {e}")
        return

    try:
        # Apply RAG chain to questions
        logging.info("Processing questions with RAG chain...")
        eval_data["rag_output"] = eval_data["question"].apply(
            lambda query: get_rag_answers(chain, query)
        )

        # Split the RAG outputs into separate columns
        rag_split = pd.DataFrame(
            eval_data["rag_output"].tolist(), columns=["rag_answer", "rag_context"]
        )

        # Add the new columns back to the original DataFrame
        eval_data = pd.concat([eval_data.drop("rag_output", axis=1), rag_split], axis=1)

        # Rename columns for clarity
        eval_data.rename(
            {
                "answer": "reference",
                "rag_answer": "response",
                "rag_context": "contexts",
                "context": "reference_contexts",
                "question": "user_input",
            },
            axis=1,
            inplace=True,
        )

        eval_data["contexts"] = eval_data["contexts"].apply(
            lambda row: [d.page_content for d in row]
        )

        # Save the updated evaluation data to a CSV file
        logging.info(f"Saving processed data to {output_file}...")
        eval_data.to_csv(output_file, index=False)
        logging.info("Data saved successfully.")
    except Exception as e:
        logging.error(f"Failed during processing or saving. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG outputs and save the results."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./evaluation_data/generated_QnA.csv",
        help="Path to the input CSV file containing questions and ground truth answers.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./chroma_index",
        help="Path to the Chroma index directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_data/evaluation_data_rag.csv",
        help="Path to save the output CSV file with evaluation results.",
    )

    args = parser.parse_args()
    logging.info("Starting RAG evaluation script...")
    main(args.input_file, args.index_path, args.output_file)
    logging.info("RAG evaluation script completed.")
