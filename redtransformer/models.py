import os
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class ModelConfig(BaseModel):
    model_name: str = "none"
    base_url: str = "none"
    api_key: str = "none"


LAMBDA_CONFIG = ModelConfig(
    base_url="https://api.lambdalabs.com/v1",
    api_key=os.getenv("LAMBDA_API_KEY", "none"),
    model_name="llama3.1-70b-instruct-berkeley",
)

GOOGLE_CONFIG = ModelConfig(
    model_name="gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY", "none"),
)

OPENAI_CONFIG = ModelConfig(
    model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "none")
)

OLLAMA_CONFIG = ModelConfig(
    model_name="llama3.1", base_url="http://localhost:11434/v1/"
)


def get_openai_client(
    model_name: Literal["openai", "ollama", "lambda"],
) -> (str, OpenAI):  # type: ignore
    match model_name:
        case "openai":
            return OPENAI_CONFIG.model_name, OpenAI()

        case "ollama":
            return OLLAMA_CONFIG.model_name, OpenAI(
                base_url=OLLAMA_CONFIG.base_url, api_key=OLLAMA_CONFIG.api_key
            )

        case "lambda":
            return LAMBDA_CONFIG.model_name, OpenAI(
                base_url=LAMBDA_CONFIG.base_url, api_key=LAMBDA_CONFIG.api_key
            )


def get_langchain_llm(
    model_name: Literal["google", "lambda", "openai", "ollama"], **kwargs
):
    match model_name:
        case "google":
            return ChatGoogleGenerativeAI(model=GOOGLE_CONFIG.model_name, **kwargs)
        case "lambda":
            return ChatOpenAI(
                model=LAMBDA_CONFIG.model_name,
                base_url=LAMBDA_CONFIG.base_url,
                api_key=LAMBDA_CONFIG.api_key,  # type: ignore
            )
        case "openai":
            return ChatOpenAI(
                model=OPENAI_CONFIG.model_name,
                api_key=OPENAI_CONFIG.api_key,  # type: ignore
            )
        case "ollama":
            return ChatOllama(model=OLLAMA_CONFIG.model_name)


def quick_talk_openai(
    query: str, model_name: Literal["openai", "ollama", "lambda"], **kwargs
):
    model_name, client = get_openai_client(model_name)

    chat_completion = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": query}], **kwargs
    )
    return chat_completion.choices[0].message.content


def quick_talk_langchain(query: str, model_name: Literal["google"], **kwargs):
    llm = get_langchain_llm(model_name)
    messages = [("human", query)]
    return llm.invoke(messages).content
