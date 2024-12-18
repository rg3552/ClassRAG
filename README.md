# CLASS-RAG

This project provides a working environment for document indexing and question-answering using Retrieval-Augmented Generation (RAG) with Chroma indexing and Cohere API. Follow the instructions below to set up and start using the application.

## Setup Instructions

### 1. Environment Setup
To set up the required environment, use the provided `environment.yml` file to create a Conda environment.

1. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

2.	**Activate the environment**:
    ```bash
    conda activate <environment_name>
    ```

### 2. API Key Configuration

You’ll need a Cohere API key to run this application.
1.	Create a .env file in the root directory of the project.
2.	Add your Cohere API key to the .env file in the following format:

```CO_API_KEY="your_cohere_api_key_here"```

### 3. Chroma Index Placement

Place the chroma_index_2 folder (containing your Chroma index files) in the root directory of the project. This index is essential for the RAG-based retrieval functionality.

Drive Link: https://drive.google.com/drive/folders/13Hdh5O7NsI52bBu3gGR0lMq-IIX5eTTP?usp=sharing

### 4. Running the Application

Once the environment is set up, API key configured, and Chroma index added, you can start using the RAG chain. Go to your project root directory and run the streamlit app!

```bash
    streamlit run main.py
```
If you encounter any issues during setup or runtime, please reach out for support.


### 5. Running Indexing and Evaluation Experiments

In the `experiments` folder, you can find notebooks to the indexing and RAGAS evaluations. The notebook is well-documented and is self explanantory.

### Additional Notes

	•	Ensure that your .env file and Chroma index files are correctly located in the root directory.
	•	For specific usage instructions and commands, refer to the code documentation.

Happy coding!
