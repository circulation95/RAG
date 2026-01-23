from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PDFPlumberLoader

def create_retriever(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever()
    return retriever
