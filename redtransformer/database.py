from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from pydantic import BaseModel


class ChromaConfig(BaseModel):
    db_path: str = "chroma_db"
    collection_name: str = "foo"
    embedding_model: str = "all-MiniLM-L6-v2"


def get_vector_store(config: ChromaConfig):
    embedding_function = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vectorstore = Chroma(
        collection_name=config.collection_name,
        embedding_function=embedding_function,
        persist_directory=config.db_path,
    )
    return vectorstore


def get_context(vectorstore, query: str, k: int = 5):
    context = ""
    results = vectorstore.similarity_search(query=query, k=k)
    for doc in results:
        context += doc.page_content
    return context
