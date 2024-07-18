import os
import subprocess
import tempfile
import uuid
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def set_environment_variables():
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GITHUB_APP_ID"] = os.getenv("GITHUB_APP_ID")
    os.environ["GITHUB_APP_PRIVATE_KEY"] = os.getenv("GITHUB_APP_PRIVATE_KEY")
    os.environ["GITHUB_BRANCH"] = "repo-rover-branch"
    os.environ["GITHUB_BASE_BRANCH"] = "main"


def set_github_repository(repo_link):
    github_url = repo_link
    repo_name = github_url.split("/")[-1]
    os.environ["GITHUB_REPOSITORY"] = repo_name


def clone_repo(repo_url, tmpdirname):
    print(f"Cloning into temporary directory: {tmpdirname}")

    # Run the git clone command
    try:
        result = subprocess.run(
            ["git", "clone", repo_url, tmpdirname], capture_output=True, text=True
        )
        # Check if the command was successful
        if result.returncode == 0:
            print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False


def load_documents(repo_path, extensions):
    documents_dict = {}
    file_type_counts = {}
    for ext in extensions:
        glob_pattern = f"**/*.{ext}"
        try:
            loader = None
            if ext == "ipynb":
                loader = NotebookLoader(
                    str(repo_path),
                    include_outputs=True,
                    max_output_length=20,
                    remove_newline=True,
                )
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata["source"]
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata["source"] = relative_path
                    doc.metadata["file_id"] = file_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue
    return documents_dict, file_type_counts


def split_documents(documents_dict, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata["file_id"] = original_doc.metadata["file_id"]
            split_doc.metadata["source"] = original_doc.metadata["source"]
        split_documents.extend(split_docs)
    return split_documents


def embed_documents(split_documents, embeddings_model):
    embeddings = []
    for split_doc in split_documents:
        content = split_doc.page_content
        embedding = embeddings_model.embed_documents([content])[
            0
        ]  # Get the embedding for the document chunk
        embeddings.append((split_doc, embedding))
        print(embeddings)
        print(len(embeddings[0][1]))
    return embeddings


def index_in_qdrant(embeddings, qdrant_client, collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "size": len(embeddings[0][1]),
            "distance": "Cosine",
        },  # Adjust based on the embedding model's vector size
    )
    points = [
        {
            "id": split_doc.metadata["file_id"],
            "vector": embedding.tolist(),
            "payload": {
                "source": split_doc.metadata["source"],
                "file_id": split_doc.metadata["file_id"],
            },
        }
        for split_doc, embedding in embeddings
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Indexed {len(points)} document chunks to Qdrant.")


def set_embeddings_model():
    # BGE from HF
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def load_and_index_files(repo_path):
    extensions = [
        "txt",
        "md",
        "markdown",
        "rst",
        "py",
        "js",
        "java",
        "c",
        "cpp",
        "cs",
        "go",
        "rb",
        "php",
        "scala",
        "html",
        "htm",
        "xml",
        "json",
        "yaml",
        "yml",
        "ini",
        "toml",
        "cfg",
        "conf",
        "sh",
        "bash",
        "css",
        "scss",
        "sql",
        "gitignore",
        "dockerignore",
        "editorconfig",
        "ipynb",
    ]

    documents_dict, file_type_counts = load_documents(repo_path, extensions)
    split_docs = split_documents(documents_dict)

    # Initialize HuggingFaceBgeEmbeddings
    embeddings_model = set_embeddings_model()

    embeddings = embed_documents(split_docs, embeddings_model)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        "localhost", port=6333
    )  # Adjust the host and port as needed

    collection_name = "repo-rover-temp-repo-store"
    index_in_qdrant(embeddings, qdrant_client, collection_name)


def set_agent_tools():
    # GitHub Toolkit
    github = GitHubAPIWrapper()
    gh_toolkit = GitHubToolkit.from_github_api_wrapper(github)
    gh_tools = gh_toolkit.get_tools()

    tools = list(gh_tools)
    tools += [
        Tool(
            name="Search",
            func=DuckDuckGoSearchRun().run,
            description="Use this tool when you need to search the web to help answer a question or solve a problem.",
        )
    ]

    return tools


def set_execution_agent(tools):
    prompt = hub.pull("repo-rover-execution-agent-prompt")

    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)
    return agent_executor


def main():
    set_environment_variables()
    repo_link = input("Enter the GitHub URL of the repository:")

    set_github_repository(repo_link)

    # new_temp_path = "/app/temp-repo"
    new_temp_path = (
        "C:\\Users\\Brett\\OneDrive\\Desktop\\firearm-research-team\\temp-repo"
    )
    os.mkdir(new_temp_path)

    clone_repo(repo_link, new_temp_path)
    load_and_index_files(new_temp_path)

    print("Files loaded and indexed successfully.")
    tools = set_agent_tools()
    print("Tools loaded successfully.")
    # agent_executor = set_execution_agent(tools)


if __name__ == "__main__":
    main()
