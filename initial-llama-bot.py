from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.base import RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
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
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from typing import List, Optional
from dotenv import load_dotenv
from typing import List
import tempfile
import logging
import asyncio
import ast
import ast
import os
import re
from typing import Union
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Literal
import operator
from typing import Annotated, List, Tuple, TypedDict
from langgraph.graph import StateGraph, START

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
    return extension_mapping.get(file_extension.lower())


def load_file(file_path):
    # Check if the file is empty
    if os.path.getsize(file_path) == 0:
        logger.info(f"Skipping empty file: {file_path}")
        return []

    file_extension = os.path.splitext(file_path)[1]
    loader_class = get_loader_class(file_extension)

    # Skip the file if its extension is not supported
    if loader_class is None:
        logger.info(f"Skipping unsupported file type: {file_path}")
        return []

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

    ignore_files = [".gitignore", "LICENSE"]

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
            # Splitting up python & regular file parsing
            # if ".py" in doc.metadata["source"]:
            #     logger.info("Splitting Python Doc")
            #     py_split_doc = split_python_document(doc)
            #     all_docs.extend(iter(py_split_doc))
            # else:
            #     logger.info("Splitting non-Python doc")
            #     if type(doc) == Document:
            #         all_docs.append(doc)
            #     else:
            #         generic_split_docs = text_splitter.split_text(doc)
            #         if type(generic_split_docs) == List:
            #             all_docs.extend(iter(generic_split_docs))
            #         else:
            #             all_docs.append(generic_split_docs)
            # Keeping all parsing the same
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
    embeddings = OllamaEmbeddings(model="bge-large")

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


async def summarize_documents(chain, documents):
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, chain.invoke, doc.page_content)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)


@tool
async def agenerate_directory_summary(documents: List[Document]) -> str:
    """Use this tool to generate summaries of directories"""

    # Define LLM
    small_llm = ChatOllama(
        model="qwen2:1.5b-instruct",
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


@tool
def generate_code(code_request: str) -> str:
    """Use this tool to generate code"""
    # Define LLM
    code_generate_llm = ChatOllama(
        model="llama3.1:8b-instruct-q2_K ",
        temperature=0.2,
        num_gpu=2,
        verbose=True,
    )
    logger.info("Set Code Generation LLM")

    code_generate_chain = create_chain(
        """You are a seasoned software developer that writes impeccable code.
    You always write code that is properly formatted according to the standards set forth by that programming language.
    You apply the necessary amount of comments that way someone new can look at your code and understand your intentions.
    The following is a request from your project manager:
    {request}
    Please write code to help them with their request
    Helpful Answer:""",
        code_generate_llm,
    )
    logger.info("Set code chain")

    code_result = code_generate_chain.invoke(code_request)
    clean_code_result = extract_code_snippet(code_result)
    clean_code_reqs = extract_code_dependencies(clean_code_result)

    return code_result, clean_code_result, clean_code_reqs


@tool
def write_code_tests(incoming_code: str) -> str:
    """Use this tool to write tests for code that is developed"""
    # Define LLM
    code_tester_llm = ChatOllama(
        model="llama3.1:8b-instruct-q2_K ",
        temperature=0,
        num_gpu=2,
        verbose=True,
    )
    logger.info("Set Code Testing LLM")

    code_tester_chain = create_chain(
        """You are a seasoned software developer that writes impeccable code tests.
Write some code tests for the following code snippet:
{snippet}
Please write an entire testing file with all code within standard code snippet indicators (```) at the beginning and end of the code block. 
Helpful Answer:""",
        code_tester_llm,
    )
    logger.info("Set Code Tester Chain")

    code_tests = code_tester_chain.invoke(incoming_code)
    clean_code_tests = extract_code_snippet(code_tests)
    clean_test_reqs = extract_code_dependencies(clean_code_tests)

    return code_tests, clean_code_tests, clean_test_reqs


@tool
def retrieve_docs(query):
    """Use this tool to retrieve documents related to the user's query"""
    docs = retriever.invoke(query)
    return docs


def extract_code_snippet(str_with_code: str) -> str:

    pattern = r"```(?:\w+)?\s*(.*?)\s*```"
    pattern2 = r"^(.*?)```"
    if result := re.search(pattern, str_with_code, re.DOTALL):
        logger.info("Extracted Code with pattern 1")
        code_snippet = result[1]
        return code_snippet
    elif result := re.search(pattern2, str_with_code, re.DOTALL):
        logger.info("Extracted Code with pattern 2")
        code_snippet = result[1]
        return code_snippet
    else:
        logger.info("Did not extract code")
        return "No code block found"


def extract_code_dependencies(code_snippet: str) -> list:
    dependencies = []
    parsed_code = ast.parse(code_snippet)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.Import):
            logger.info("Extracted Import Dependency")
            dependencies.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            logger.info(f"Extracted {node.module} Dependency")
            dependencies.append(node.module)

    return dependencies


def set_environment_variables():
    load_dotenv()
    logger.info("Loading Langsmith environment variables")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    logger.info("Langsmith environment variables loaded")


def set_execution_agent():
    tools = [
        generate_code,
        write_code_tests,
        agenerate_directory_summary,
        retrieve_docs,
    ]
    llm = ChatOllama(
        model="llama3.1:8b-instruct-q2_K",
        temperature=0.5,
        num_gpu=2,
        verbose=True,
    )
    executor = create_react_agent(llm, tools, messages_modifier=prompt)
    return executor


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


def set_planning_agent():
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    planner = planner_prompt | ChatOllama(
        model="llama3.1:8b-instruct-q2_K", temperature=0
    ).with_structured_output(pydantic_object=Plan)

    return planner


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


async def execute_step(state: PlanExecute):
    agent_executor = set_execution_agent()
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": (task, agent_response["messages"][-1].content),
    }


async def plan_step(state: PlanExecute):
    planner = set_planning_agent()
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    replanner = set_replanning_agent()
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


def set_replanning_agent():

    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )

    replanner = replanner_prompt | ChatOllama(
        model="llama3.1:8b-instruct-q2_K", temperature=0
    ).with_structured_output(Act)

    return replanner


def set_agent_graph():
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    return app


if __name__ == "__main__":

    app = set_agent_graph()
    config = {"recursion_limit": 50}
    inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

    # planner = set_planning_agent()

    # response = planner.invoke(
    #     {
    #         "messages": [
    #             ("user", "what is the hometown of the current Australia open winner?")
    #         ]
    #     }
    # )
    # print(response)

    # set_environment_variables()

    # directory_path = input("Enter the directory path to process:")
    # directories_to_check = [directory_path]
    # ignore_input = input(
    #     "Enter any sub-directories to ignore (comma-separated, press Enter for none): "
    # )

    # temp_dir = create_temp_directory()
    # logger.info(f"Created temporary directory: {temp_dir}")

    # ignore_directories = (
    #     [dir.strip() for dir in ignore_input.split(",")] if ignore_input else []
    # )

    # # Add common virtual environment directory names if not specified by the user
    # common_venv_names = ["venv", "env", ".venv", ".env", ".git", "__pycache__"]
    # ignore_directories.extend(
    #     [venv for venv in common_venv_names if venv not in ignore_directories]
    # )

    # try:
    #     # Process documents and create FAISS index (do this only when new documents are added)
    #     vectorstore, split_docs = process_and_embed_documents(
    #         directories_to_check, temp_dir, ignore_directories
    #     )
    #     logger.info(f"Created FAISS index at: {temp_dir}")

    #     # Later, when you need to use the embeddings:
    #     loaded_vectorstore = load_faiss_index(temp_dir)
    #     logger.info("Loading FAISS Index into memory")

    #     retriever = loaded_vectorstore.as_retriever(
    #         search_type="mmr", search_kwargs={"k": 10}
    #     )
    #     logger.info("Set FAISS Vector Store as a Retriever")

    #     generated_response, generated_code, code_dependencies = generate_code(
    #         "Create a rock, paper, scissors program"
    #     )
    #     print("RESPONSE------------")
    #     print(generated_response)
    #     print("CODE-----------------")
    #     print(generated_code)
    #     print("DEPENDENCIES---------------")
    #     print(code_dependencies)

    #     generated_test_response, generated_code_tests, test_dependencies = (
    #         write_code_tests(generated_code)
    #     )
    #     print("RESPONSE------------")
    #     print(generated_test_response)
    #     print("CODE-----------------")
    #     print(generated_code_tests)
    #     print("DEPENDENCIES---------------")
    #     print(test_dependencies)

    #     #     summary = asyncio.run(generate_directory_summary(split_docs))
    #     #     logger.info("Created directory summary")

    #     #     print(
    #     #         "I have finished preparing everything related to understanding that directory. If you want to exit type '/bye' otherwise...."
    #     #     )
    #     #     event = input("What would you like to do?")
    #     #     while True:
    #     #         print(summary)
    #     #         event = input("Anything else?")
    #     #         if event.lower() == "/bye":
    #     #             print("Bye Bye!")
    #     #             break
    #     # except Exception as e:
    #     # logger.error(f"An error occurred: {str(e)}")
    # finally:
    #     cleanup_temp_directory(temp_dir)
    #     logger.info(f"Cleaned up temporary directory: {temp_dir}")
