from langchain_core.tools import tool



# @tool turns our Python function into a LangChain tool.

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Could not calculate the expression."