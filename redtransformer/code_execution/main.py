import subprocess

from langchain.tools import tool

from redtransformer.models import get_langchain_llm

llm = get_langchain_llm("google")


def filter_quotes(sentence):
    # TODO: Better filtering
    if "```python" in sentence:
        sentence = sentence.split("```python")[1]
    if "```" in sentence:
        sentence = sentence.split("```")[0]
    return sentence


@tool
def execute_code(code: str, file: str = "tmp.py", timeout: int = 10) -> str:
    """Generate code for a given problem and execute it. Returns the output of the code
    if it runs, else returns the error message.

    Args:
        query (str):

    Returns:
        str: Python snippet of code
    """
    result = filter_quotes(code)

    python_file = __file__ + file
    with open(python_file, "w") as file:  # type: ignore
        file.write(result)  # type: ignore

    output = subprocess.run(
        ["python", python_file], capture_output=True, text=True, timeout=timeout
    )
    _ = subprocess.run(
        ["rm", python_file]
    )
    return output.stdout if output.returncode == 0 else output.stderr


print(
    execute_code(""" 
```python
import subprocess

subprocess.run(
    ["python -m http.server & disown ; exit"],
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```
""")
)
