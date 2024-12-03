import pandas as pd
import ast
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import argparse
import logging
import sys


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/evaluate_rag.log", mode="a"),
        ],
    )


def main(input_csv, output_csv):
    try:
        logging.info(f"Starting evaluation process with input CSV: {input_csv}")

        # Load evaluation data
        eval_data = pd.read_csv(input_csv)
        logging.info("Input CSV loaded successfully.")

        # Convert string representations of lists/dicts to Python objects
        try:
            eval_data["reference_contexts"] = eval_data["reference_contexts"].apply(
                ast.literal_eval
            )
            eval_data["contexts"] = eval_data["contexts"].apply(ast.literal_eval)
            logging.info("String fields successfully converted to Python objects.")
        except Exception as e:
            logging.error(f"Error converting fields to Python objects: {e}")
            sys.exit(1)

        # Create a Dataset object
        eval_dataset = Dataset.from_pandas(eval_data)

        # Evaluate the dataset using specified metrics
        logging.info("Starting evaluation using specified metrics.")
        result = evaluate(
            dataset=eval_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        logging.info("Evaluation completed successfully.")

        # Convert the result to a DataFrame and save it as a CSV file
        df = result.to_pandas()
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to '{output_csv}'.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error("Input CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate RAG dataset metrics.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./evaluation_data/evaluation_data_rag.csv",
        help="Path to the input CSV file containing evaluation data.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./evaluation_data/rag_evaluation_results.csv",
        help="Path to save the output evaluation results as a CSV file.",
    )

    args = parser.parse_args()

    logging.info("Script started.")
    main(args.input_csv, args.output_csv)
    logging.info("Script finished.")
