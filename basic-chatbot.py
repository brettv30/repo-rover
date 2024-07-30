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
from langgraph.graph import StateGraph, START
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
{plan_str}\n\nYou are tasked with executing step 1, {task}."""
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
    return "__end__" if "response" in state and state["response"] else "agent"


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


def set_embeddings_model():
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    return embedder


async def set_agent_tools():
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


async def set_planner_agent():
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


async def set_replanner_agent():
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
