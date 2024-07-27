import os
import asyncio
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import (
    GitLoader,
)
from langgraph.graph import StateGraph, START
from langchain_nomic import NomicEmbeddings
import operator
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict, Union, Literal

# load_dotenv()


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


async def execute_step(state: PlanExecute):
    executor = set_execution_agent()
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await executor.ainvoke({"messages": [("user", task_formatted)]})
    return {
        "past_steps": (task, agent_response["messages"][-1].content),
    }


async def plan_step(state: PlanExecute):
    planner = set_planner_agent()
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    replanner = set_replanner_agent()
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


def read_secret(secret_name):
    path = os.path.join("/run/secrets", secret_name)
    if os.path.exists(path):
        with open(path, "r") as secret_file:
            return secret_file.read().strip()
    return None


def set_environment_variables():
    os.environ["LANGCHAIN_API_KEY"] = read_secret("langchain_api_key")
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # os.environ["NOMIC_API_KEY"] = os.getenv("NOMIC_API_KEY")
    # os.environ["GITHUB_APP_ID"] = int(read_secret("github_app_id"))
    os.environ["GITHUB_APP_PRIVATE_KEY"] = read_secret("github_app_private_key")
    os.environ["GITHUB_REPOSITORY"] = read_secret("github_repo")
    os.environ["ACTIVE_BRANCH"] = "repo-rover-branch"
    os.environ["GITHUB_BASE_BRANCH"] = "main"


def set_github_repository(repo_link):
    github_url = repo_link
    repo_name = github_url.split("/")[-1]
    os.environ["GITHUB_REPOSITORY"] = repo_name


def set_github_base_branch(branch_name):
    os.environ["GITHUB_BASE_BRANCH"] = branch_name


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
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    return embedder


def load_and_index_files(repo_link, repo_path):

    docs = load_documents(repo_link, repo_path)
    print(docs)
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
    github = GitHubAPIWrapper(
        github_app_id=int(read_secret("github_app_id")),
        github_app_private_key=os.getenv("GITHUB_APP_PRIVATE_KEY"),
        active_branch=os.getenv("ACTIVE_BRANCH"),
        github_base_branch=os.getenv("GITHUB_BASE_BRANCH"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
    )
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


def set_planner_agent():
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
        model="llama3.1:8b-instruct-q2_K",
        temperature=0,
        num_gpu=1,
        verbose=True,
    ).with_structured_output(Plan)

    return planner


def set_replanner_agent():
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
        model="llama3.1:8b-instruct-q2_K",
        temperature=0,
        num_gpu=1,
        verbose=True,
    ).with_structured_output(Act)

    return replanner


def set_execution_agent():
    prompt = hub.pull("wfh/react-agent-executor")

    # Choose the LLM that will drive the agent
    llm = ChatOllama(
        model="phi3:3.8b-mini-128k-instruct-q3_K_S", num_gpu=1, verbose=True
    )
    tools = set_agent_tools()
    executor = create_react_agent(llm, tools, messages_modifier=prompt)
    return executor


async def main():
    set_environment_variables()
    print("Set Environment Variables")

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

    config = {"recursion_limit": 50}
    inputs = {"input": "What GitHub repository do you have access to?"}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    asyncio.run(main())
