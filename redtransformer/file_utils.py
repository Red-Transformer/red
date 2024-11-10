import os
from typing import List, Optional

import aiohttp
import pymupdf  # type: ignore
import pymupdf4llm  # type: ignore
from jinja2 import Environment, FileSystemLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_template(template_name: str, **kwargs) -> str:
    """
    Load a Jinja template an fill it with the kwargs.
    """
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    return template.render(kwargs)


def load_documents(folder: str, pattern: str = "**/*.md") -> List[Document]:
    """
    Load documents from folder using Langchain functions.
    """
    loader = DirectoryLoader(folder, glob=pattern)
    documents = loader.load()
    return documents


def documents_to_splits(
    documents: List[Document], chunk_overlap: int = 200, chunk_size: int = 1000, *args
):
    """
    Converts Langchain documents to text splits, which can be used to create a RAG database.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap, chunk_size=chunk_size
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(splits)} chunks.")

    return splits


def pdf_to_text(file_path, num_pages: Optional[int] = None, min_size: int = 100) -> str:
    """
    Extract text from the PDF file using pymupdf4llm, falling back to pymupdf and pypdf if necessary.
    """
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(file_path)
        else:
            reader = PdfReader(file_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(file_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(file_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(file_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short") from e

    return text


async def download_file(file_link: str, dest_path: str) -> str:
    """
    Download a file from a given link to a specified destination path, not necessarily from ArXiv.
    """
    async with aiohttp.ClientSession() as session, session.get(file_link) as response:
        if response.status != 200:
            raise Exception(f"Error downloading file: {response.status}")
        with open(dest_path, "wb") as f:
            f.write(await response.read())
        return dest_path
