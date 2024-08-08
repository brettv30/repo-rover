from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import asyncio
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import tempfile
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
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
import ast
import re
from typing import List
from langchain.text_splitter import TextSplitter
from langchain.schema import Document

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
        ".md": UnstructuredMarkdownLoader,
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

    ignore_files = [".gitignore", "LICENSE"]

    documents = []
    for root, dirs, files in os.walk(directory_path, topdown=True):
        # Remove ignored directories from dirs to prevent os.walk from traversing them
        dirs[:] = [d for d in dirs if d not in ignore_directories]
        files[:] = [f for f in files if f not in ignore_files]

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


def load_file(file_path):
    file_extension = os.path.splitext(file_path)[1]
    loader_class = get_loader_class(file_extension)
    logger.info(f"Loading {file_path} with {loader_class.__name__}")
    try:
        loader = loader_class(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return []


def load_documents_sync(
    directory_path: str, ignore_directories: Optional[List[str]] = None
):
    if ignore_directories is None:
        ignore_directories = []

    ignore_files = [".gitignore", "LICENSE", "log.txt"]

    documents = []
    with ThreadPoolExecutor() as executor:
        file_load_futures = []
        for root, dirs, files in os.walk(directory_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in ignore_directories]
            files[:] = [f for f in files if f not in ignore_files]

            if any(
                ignore_dir in root.split(os.sep) for ignore_dir in ignore_directories
            ):
                logger.info(f"Skipping ignored directory: {root}")
                continue

            for file in files:
                file_path = os.path.join(root, file)

                if any(
                    ignore_dir in file_path.split(os.sep)
                    for ignore_dir in ignore_directories
                ):
                    logger.info(f"Skipping file in ignored directory: {file_path}")
                    continue

                file_load_futures.append(executor.submit(load_file, file_path))

        for future in file_load_futures:
            documents.extend(future.result())

    return documents


async def load_documents(
    directory_path: str, ignore_directories: Optional[List[str]] = None
):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        documents = await loop.run_in_executor(
            executor, load_documents_sync, directory_path, ignore_directories
        )
    return documents


async def load_all_documents(
    directories: List[str], ignore_directories: Optional[List[str]] = None
):
    tasks = [load_documents(directory, ignore_directories) for directory in directories]
    all_documents = await asyncio.gather(*tasks)
    # Flatten the list of lists
    return [doc for sublist in all_documents for doc in sublist]


class PythonDocumentSplitter(TextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, document: Document) -> List[Document]:
        text = document.page_content
        source = document.metadata["source"]
        documents = []

        # Parse the Python code
        tree = ast.parse(text)

        if global_chunk := "".join(
            ast.get_source_segment(text, node) + "\n"
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
        ):
            documents.append(
                Document(
                    page_content=global_chunk.strip(),
                    metadata={"source": source, "type": "global"},
                )
            )

        # Extract classes and their methods
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_content = ast.get_source_segment(text, node)
                documents.append(
                    Document(
                        page_content=class_content,
                        metadata={
                            "source": source,
                            "type": "class",
                            "name": node.name,
                        },
                    )
                )

                # Extract methods within the class
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_content = ast.get_source_segment(text, child)
                        documents.append(
                            Document(
                                page_content=method_content,
                                metadata={
                                    "source": source,
                                    "type": "method",
                                    "class": node.name,
                                },
                            )
                        )

        # Extract standalone functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_content = ast.get_source_segment(text, node)
                documents.append(
                    Document(
                        page_content=function_content,
                        metadata={
                            "source": source,
                            "type": "function",
                            "name": node.name,
                        },
                    )
                )

        # Extract main execution block
        main_pattern = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:')
        if main_match := main_pattern.search(text):
            main_block = text[main_match.start() :]
            documents.append(
                Document(
                    page_content=main_block,
                    metadata={"source": source, "type": "main_execution_block"},
                )
            )

        return documents


def split_python_document(document: Document) -> List[Document]:
    splitter = PythonDocumentSplitter()
    return splitter.split_text(document)


def split_multiple_python_documents(documents: List[Document]) -> List[Document]:
    all_split_documents = []
    for doc in documents:
        all_split_documents.extend(split_python_document(doc))
    return all_split_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_docs = []
    for doc in documents:
        if len(doc.page_content) > 5000:
            if ".py" in doc.metadata["source"]:
                logger.info("Splitting Python Doc")
                py_split_doc = split_python_document(doc)
                all_docs.extend(iter(py_split_doc))
            else:
                logger.info("Splitting non-Python doc")
                if type(doc) == Document:
                    all_docs.append(doc)
                else:
                    generic_split_docs = text_splitter.split_text(doc)
                    if type(generic_split_docs) == List:
                        all_docs.extend(iter(generic_split_docs))
                    else:
                        all_docs.append(generic_split_docs)
        else:
            all_docs.append(doc)

    return all_docs


def process_and_embed_documents(
    directory_path,
    index_path,
    ignore_directories: Optional[List[str]] = None,
    chunk_size=1000,
    chunk_overlap=200,
):
    try:
        raw_documents = asyncio.run(
            load_all_documents(directory_path, ignore_directories)
        )
        logger.info(f"Loaded Documents to {directory_path}")
        split_docs = split_documents(raw_documents, chunk_size, chunk_overlap)
        logger.info("Split Documents Completed")

        # Embed documents and create FAISS index
        vectorstore = embed_documents(split_docs)
        if vectorstore is None:
            raise ValueError("Failed to create FAISS vectorstore")

        logger.info("Created FAISS vectorstore")

        # Save the FAISS index
        save_faiss_index(vectorstore, index_path)
        logger.info(f"Saved FAISS Index at {index_path}")

        return vectorstore, split_docs
    except Exception as e:
        logger.error(f"An error occurred in process_and_embed_documents: {str(e)}")
        raise


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def embed_documents(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create a FAISS index from the documents
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


# def embed_documents(documents):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     flat_docs = flatten_list(documents)
#     logger.info(f"Total documents to embed: {len(flat_docs)}")

#     all_embeddings = []

#     for i, doc in enumerate(flat_docs):
#         if not isinstance(doc, Document):
#             logger.error(f"Unexpected document type at index {i}: {type(doc)}")
#             continue

#         # Debug: Check document content and metadata
#         logger.debug(
#             f"Document index: {i}, content: {doc.page_content[:100]}..."
#         )  # Log first 100 characters
#         logger.debug(f"Document metadata: {doc.metadata}")

#         try:
#             embedding = embeddings.embed_query(doc.page_content)
#             logger.debug(f"Embedding shape at index {i}: {np.shape(embedding)}")
#             all_embeddings.append(embedding)
#         except Exception as e:
#             logger.error(f"Error embedding document at index {i}: {str(e)}")

#     # Check if all embeddings have the same shape
#     try:
#         # Ensure all embeddings are numpy arrays and log their shapes
#         embedding_shapes = [np.shape(emb) for emb in all_embeddings]
#         logger.info(f"All embedding shapes: {embedding_shapes}")

#         # Convert to a numpy array
#         all_embeddings = np.array(all_embeddings)
#         logger.info(f"Embeddings shape: {all_embeddings.shape}")
#     except Exception as e:
#         logger.error(f"Error in embeddings shape: {str(e)}")

#     # Create a FAISS index from the embeddings
#     try:
#         vectorstore = FAISS.from_embeddings(all_embeddings, flat_docs)
#     except Exception as e:
#         logger.error(f"Error creating FAISS index: {str(e)}")
#         return None

#     return vectorstore


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


async def summarize_documents(chain, documents):
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, chain.invoke, doc.page_content)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)


async def generate_directory_summary(documents: List[Document]) -> str:

    # Define LLM
    small_llm = ChatOllama(
        model="qwen2:0.5b-instruct",
        temperature=0.2,
        num_gpu=2,
        verbose=True,
    )
    logger.info("Set small summarizer LLM")

    # Define LLM
    big_llm = ChatOllama(
        model="gemma2:2b-instruct-q8_0 ",
        temperature=0.5,
        num_gpu=2,
        verbose=True,
    )
    logger.info("Set big summarizer LLM")

    map_chain = create_chain(
        """You are a concise summarizer. You write concise summaries of documents that extract only the main parts of the document. 
    The following is a document
    {docs}
    Please identify the main theme or themes of this document. Be concise.
    Helpful Answer:""",
        small_llm,
    )
    logger.info("Set map chain")

    reduce_chain = create_chain(
        """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. Each summary is about a different aspect of files in a directory so always reference 'the directory' in your response, never 'the summaries.'
    Helpful Answer:""",
        big_llm,
    )
    logger.info("Set reduce chain")

    # Process documents in batches if necessary
    batch_size = 10  # Adjust based on your hardware capabilities
    individual_summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        summaries = await summarize_documents(map_chain, batch)
        individual_summaries.extend(summaries)

    # individual_summaries = await summarize_documents(map_chain, documents)
    final_summary = reduce_chain.invoke(individual_summaries)

    return final_summary


def create_chain(prompt_value: str, llm: ChatOllama) -> RunnableSequence:
    # Map
    prompt = PromptTemplate.from_template(prompt_value)
    result = prompt | llm | StrOutputParser()

    return result


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


def set_environment_variables():
    load_dotenv()
    logger.info("Loading Langsmith environment variables")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    logger.info("Langsmith environment variables loaded")


if __name__ == "__main__":

    set_environment_variables()

    directories_to_check = []

    directory_path = input("Enter the directory path to process:")
    directories_to_check.append(directory_path)
    ignore_input = input(
        "Enter any sub-directories to ignore (comma-separated, press Enter for none): "
    )

    temp_dir = create_temp_directory()
    logger.info(f"Created temporary directory: {temp_dir}")

    ignore_directories = (
        [dir.strip() for dir in ignore_input.split(",")] if ignore_input else []
    )

    # Add common virtual environment directory names if not specified by the user
    common_venv_names = ["venv", "env", ".venv", ".env", ".git", "__pycache__"]
    ignore_directories.extend(
        [venv for venv in common_venv_names if venv not in ignore_directories]
    )

    try:
        # Process documents and create FAISS index (do this only when new documents are added)
        vectorstore, split_docs = process_and_embed_documents(
            directories_to_check, temp_dir, ignore_directories
        )
        logger.info(f"Created FAISS index at: {temp_dir}")

        # Later, when you need to use the embeddings:
        loaded_vectorstore = load_faiss_index(temp_dir)
        logger.info("Loading FAISS Index into memory")

        retriever = loaded_vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )
        logger.info("Set FAISS Vector Store as a Retriever")

        summary = asyncio.run(generate_directory_summary(split_docs))
        logger.info("Created directory summary")

        print(
            "I have finished preparing everything related to understanding that directory. If you want to exit type '/bye' otherwise...."
        )
        event = input("What would you like to do?")
        while True:
            print("mama we made it")
            event = input("Anything else?")
            if event.lower() == "/bye":
                print("Bye Bye!")
                break
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
