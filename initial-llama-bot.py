from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import tempfile
from typing import List
from typing_extensions import TypedDict
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredFileLoader,
    PythonLoader,
)
import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_user(user_id: int, addresses: List) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id: (int) the user ID.
        addresses: Previous addresses.
    """
    return True


def get_loader_class(file_extension):
    extension_mapping = {
        ".txt": TextLoader,
        ".py": PythonLoader,
        ".pdf": PDFMinerLoader,
        ".csv": CSVLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".html": UnstructuredHTMLLoader,
    }
    return extension_mapping.get(file_extension.lower(), UnstructuredFileLoader)


def load_documents(directory_path: str, ignore_directories: Optional[List[str]] = None):
    if ignore_directories is None:
        ignore_directories = []

    documents = []
    for root, dirs, files in os.walk(directory_path, topdown=True):
        # Remove ignored directories from dirs to prevent os.walk from traversing them
        dirs[:] = [d for d in dirs if d not in ignore_directories]

        # Check if the current directory should be ignored
        if any(ignore_dir in root.split(os.sep) for ignore_dir in ignore_directories):
            logger.info(f"Skipping ignored directory: {root}")
            continue  # Skip this directory

        for file in files:
            file_path = os.path.join(root, file)

            # Double-check that the file is not in an ignored directory
            if any(
                ignore_dir in file_path.split(os.sep)
                for ignore_dir in ignore_directories
            ):
                logger.info(f"Skipping file in ignored directory: {file_path}")
                continue

            file_extension = os.path.splitext(file)[1]
            loader_class = get_loader_class(file_extension)
            logger.info(f"Loading {file_path} with {loader_class.__name__}")
            try:
                loader = loader_class(file_path)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def process_and_embed_documents(
    directory_path,
    index_path,
    ignore_directories: Optional[List[str]] = None,
    chunk_size=1000,
    chunk_overlap=200,
):
    raw_documents = load_documents(directory_path, ignore_directories)
    print("Loaded Documents")
    split_docs = split_documents(raw_documents, chunk_size, chunk_overlap)
    print("Split Documents")

    # Embed documents and create FAISS index
    vectorstore = embed_documents(split_docs)
    print("Created FAISS vectorstore")

    # Save the FAISS index
    save_faiss_index(vectorstore, index_path)
    print("Saved FAISS Index")

    return vectorstore


def embed_documents(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create a FAISS index from the documents
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def save_faiss_index(vectorstore, index_path):
    # Save the FAISS index
    vectorstore.save_local(index_path)


def load_faiss_index(index_path):
    # Initialize the embedding model (must be the same as used for creation)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load the FAISS index
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    return vectorstore


def create_temp_directory():
    return tempfile.mkdtemp()


def cleanup_temp_directory(temp_dir):
    import shutil

    shutil.rmtree(temp_dir)


prompt = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search

You have access to the following functions:

Use the function 'validate_user' to: validate user using historical addresses
{{
  "name": "validate_user",
  "description": "validate user using historical addresses",
  "parameters": {{
    "user_id": {{
      "param_type": "int",
      "description": "User ID",
      "required": true
    }},
    "addresses": {{
       "param_type": "List",
       "description": "The user's previous addresses.",
       "required": true
    }}
  }}
}}

If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line"
- Always add your sources when using search results to answer the user query

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION: {question} 
 <|eot_id|><|start_header_id|>assistant<|end_header_id|>                    
    """
)

if __name__ == "__main__":

    directory_path = input("Enter the directory path to process:")
    ignore_input = input(
        "Enter directories to ignore (comma-separared, press Enter for none): "
    )
    # directory_path = ""

    temp_dir = create_temp_directory()
    logger.info(f"Created temporary directory: {temp_dir}")

    ignore_directories = (
        [dir.strip() for dir in ignore_input.split(",")] if ignore_input else []
    )

    # Add common virtual environment directory names if not specified by the user
    common_venv_names = ["venv", "env", ".venv", ".env", ".git"]
    ignore_directories.extend(
        [venv for venv in common_venv_names if venv not in ignore_directories]
    )

    try:
        # Process documents and create FAISS index (do this only when new documents are added)
        vectorstore = process_and_embed_documents(
            directory_path, temp_dir, ignore_directories
        )
        logger.info(f"Created FAISS index at: {temp_dir}")

        # Later, when you need to use the embeddings:
        loaded_vectorstore = load_faiss_index(temp_dir)
        print("Loaded FAISS Index")

        # Now you can use loaded_vectorstore for similarity search, etc.
        query = "How are Key, Query, and Value defined in the documents?"
        results = loaded_vectorstore.similarity_search(query)
        print(results)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        cleanup_temp_directory(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    # llm = ChatOllama(
    #     model="llama3.1:8b-instruct-q2_K",
    #     temperature=0.1,
    #     num_gpu=1,
    #     verbose=True,
    # ).bind_tools([validate_user])

    # chain = prompt | llm

    # response = chain.invoke(
    #     {
    #         "question": "Could you validate user 123? They previously lived at 123 Fake St in Boston MA and 234 Pretend Boulevard in Houston TX."
    #     }
    # )

    # print(response)
    # print(response.tool_calls)
