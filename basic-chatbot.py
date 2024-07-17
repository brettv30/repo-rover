import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()


def set_environment_variables():
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GITHUB_APP_ID"] = os.getenv("GITHUB_APP_ID")
    os.environ["GITHUB_APP_PRIVATE_KEY"] = os.getenv("GITHUB_APP_PRIVATE_KEY")
    os.environ["GITHUB_REPOSITORY"] = os.getenv("GITHUB_REPOSITORY")
    os.environ["GITHUB_BRANCH"] = os.getenv("GITHUB_BRANCH")
    os.environ["GITHUB_BASE_BRANCH"] = os.getenv("GITHUB_BASE_BRANCH")


def set_embeddings():
    # BGE from HF
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def set_github_repository(repo_link):
    github_url = repo_link
    repo_name = github_url.split("/")[-1]


def set_vector_store():
    # Qdrant
    client = QdrantClient("localhost", port=6333)
    embeddings = set_embeddings()
    vector_store = Qdrant(
        client=client,
        collection_name="repo-rover-temp-repo-store",
        embeddings=embeddings,
    )
    return vector_store


def set_agent_tools():
    # GitHub Toolkit
    github = GitHubAPIWrapper()
    gh_toolkit = GitHubToolkit.from_github_api_wrapper(github)
    gh_tools = gh_toolkit.get_tools()

    tools = []

    for tool in gh_tools:
        tools.append(tool)

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


if __name__ == "__main__":
    set_environment_variables()
    tools = set_agent_tools()
    agent_executor = set_execution_agent(tools)
    agent_executor.invoke("who is the winnner of the us open")
