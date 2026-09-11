from langchain_groq import ChatGroq
from tools import calculator
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [calculator]

llm_with_tools = llm.bind_tools(tools)


def run_agent(question):

    response = llm_with_tools.invoke(question)

    return response