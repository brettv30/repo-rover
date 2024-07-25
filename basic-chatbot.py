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
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    NotebookLoader,
    GitLoader,
    GitHubIssuesLoader,
    GithubFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def set_environment_variables():
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GITHUB_APP_ID"] = os.getenv("GITHUB_APP_ID")
    os.environ["GITHUB_APP_PRIVATE_KEY"] = os.getenv("GITHUB_APP_PRIVATE_KEY")
    os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = os.getenv(
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    )
    os.environ["GITHUB_BRANCH"] = "repo-rover-branch"


def set_github_repository(repo_link):
    github_url = repo_link
    repo_name = github_url.split("/")[-1]
    os.environ["GITHUB_REPOSITORY"] = repo_name


def set_github_base_branch(branch_name):
    os.environ["GITHUB_BASE_BRANCH"] = branch_name


def load_github_files(repo_url):
    try:
        loader = GithubFileLoader(
            repo=repo_url,
            access_token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        )
        return loader.load()
    except Exception as e:
        print(f"An error occurred while loading GitHub Repo files: {e}")
        return False


def load_github_issues(repo_url):
    try:
        print("Grabbing issues and Pull Requests from {repo_url}")
        loader = GitHubIssuesLoader(
            repo="brettv30/Call-of-Duty-Game-Predictions",
            access_token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        )
        return loader.load()
    except Exception as e:
        print(f"An error occurred while loading GitHub Issues: {e}")
        return False


def load_documents(repo_url, repo_path):
    try:
        print(f"Cloning {repo_url} into temporary directory: {repo_path}")
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=repo_path,
            branch=os.environ["GITHUB_BASE_BRANCH"],
        )
        return loader.load()
    except Exception as e:
        print(f"Failed to clone repository and load repo contents. Error: {e}")
        return False


def split_documents(documents_dict, chunk_size=800, chunk_overlap=100):
    return True


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
    if qdrant_client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
    else:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings[0][1]),
                distance=models.Distance.COSINE,
            ),  # Adjust based on the embedding model's vector size
        )

    points = [
        models.PointStruct(
            id=split_doc.metadata["file_id"],
            vector=embedding.tolist(),
            payload={
                "source": split_doc.metadata["source"],
                "file_id": split_doc.metadata["file_id"],
            },
        )
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


def load_and_index_files(repo_link, repo_path):

    docs = load_documents(repo_link, repo_path)
    # docs = load_github_files(repo_link)
    # issues = load_github_issues(repo_link)
    print(docs)
    # print("-----------------------------")
    # print(issues)
    split_docs = split_documents(docs)

    # Initialize HuggingFaceBgeEmbeddings
    embeddings_model = set_embeddings_model()

    embeddings = embed_documents(split_docs, embeddings_model)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        "localhost", port=6333, api_key=os.getenv("QDRANT_API_KEY")
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
    branch = input("Enter the name of the base branch:")

    set_github_repository(repo_link)
    set_github_base_branch(branch)

    # new_temp_path = "/app/temp-repo"
    new_temp_path = (
        "C:\\Users\\Brett\\OneDrive\\Desktop\\firearm-research-team\\temp-repo"
    )
    os.mkdir(new_temp_path)

    load_and_index_files(repo_link, new_temp_path)

    print("Files loaded and indexed successfully.")
    tools = set_agent_tools()
    print("Tools loaded successfully.")
    # agent_executor = set_execution_agent(tools)


if __name__ == "__main__":
    main()
