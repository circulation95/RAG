import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS


def init_retriever(db_index="LANGCHAIN_DB_INDEX", fetch_k=30, top_n=8):
    # Embeddings 설정
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # 저장된 DB 로드
    langgraph_db = FAISS.load_local(
        db_index, embeddings, allow_dangerous_deserialization=True
    )
    # retriever 생성
    code_retriever = langgraph_db.as_retriever(search_kwargs={"k": fetch_k})

    # JinaRerank 설정
    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=code_retriever
    )
    return compression_retriever
