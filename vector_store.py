from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_vector_store(pages, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for i, page in enumerate(pages):
        chunks = splitter.split_text(page)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"page_number": i + 1}))
    embeddings = OpenAIEmbeddings(api_key=api_key)
    store = FAISS.from_documents(docs, embeddings)
    return store
