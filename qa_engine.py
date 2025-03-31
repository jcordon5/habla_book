from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document


class FilteredRetriever(BaseRetriever):
    def __init__(self, base_retriever, max_page: int):
        super().__init__()
        self._base_retriever = base_retriever
        self._max_page = max_page

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._base_retriever.get_relevant_documents(query)
        return [doc for doc in docs if doc.metadata.get("page_number", 0) <= self._max_page]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self._base_retriever.aget_relevant_documents(query)
        return [doc for doc in docs if doc.metadata.get("page_number", 0) <= self._max_page]


def create_qa_chain(store, max_page, api_key):
    retriever = store.as_retriever()
    filtered_retriever = FilteredRetriever(retriever, max_page)
    llm = ChatOpenAI(temperature=0.2, api_key=api_key)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=filtered_retriever,
        return_source_documents=True
    )
    return chain
