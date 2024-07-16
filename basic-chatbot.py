import os
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import TextRequestsWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from ionic_langchain.tool import Ionic, IonicTool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import ShellTool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

load_dotenv()


def set_environment_variables():
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Research Tools
ddg_search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivAPIWrapper()
requests = TextRequestsWrapper()
pubmed = PubmedQueryRun()

# Niche Tools
ionic_tool = IonicTool().tool()
youtube = YouTubeSearchTool()
yahoo_finance = YahooFinanceNewsTool()

# Code/File Management Tools
py_code_executor = PythonREPLTool()
file_management_tool = FileManagementToolkit()
shell_tool = ShellTool()

# GitHub Toolkit
github = GitHubAPIWrapper()
gh_toolkit = GitHubToolkit.from_github_api_wrapper(github)
gh_tools = gh_toolkit.get_tools()
