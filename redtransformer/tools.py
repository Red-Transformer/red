from typing import Optional

import aiohttp
import pymupdf  # type: ignore
import pymupdf4llm  # type: ignore
from pypdf import PdfReader


def pdf_to_text(file_path, num_pages: Optional[int] = None, min_size: int = 100):
    """
    Extract text from the PDF file using pymupdf4llm, falling back to pymupdf and pypdf if necessary.

    :param num_pages:
    :param min_size:
    :return:
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


async def download_file(file_link: str, dest_path: str):
    """
    Download a file from a given link to a specified destination path, not necessarily from ArXiv.

    :param file_link:
    :param dest_path:
    :return:
    """
    async with aiohttp.ClientSession() as session, session.get(file_link) as response:
        if response.status != 200:
            raise Exception(f"Error downloading file: {response.status}")
        with open(dest_path, "wb") as f:
            f.write(await response.read())
        return dest_path
