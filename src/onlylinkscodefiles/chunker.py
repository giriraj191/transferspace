from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import warnings
warnings.simplefilter("ignore")


def chunk_doc(document, chunk_size, chunk_overlap):
    doc_type = document.metadata.get("type", "")
    
    if doc_type == "text":
        chunk_size = chunk_size
        chunk_overlap = chunk_overlap
        separators = ["\n\n", "\n", ".", " "]
    else: 
        chunk_size = 1200
        chunk_overlap = 100
        separators = ["```", "~~~", "\n\n", "\n-\n", "|", 
                      "<table>", "</table>", "<img", "![", 
                      "<figure>", "</figure>"]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,  
        separators=separators
    )

    text_chunks = splitter.split_text(document.page_content)
    
    return [Document(page_content=chunk, metadata=document.metadata) for chunk in text_chunks]


def chunker(documents, chunk_size, chunk_overlap):
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(chunk_doc(doc, chunk_size, chunk_overlap))
    return chunked_documents