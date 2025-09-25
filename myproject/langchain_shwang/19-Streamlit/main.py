import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_teddynote import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
    )

    option = st.selectbox(
        "프롬프트 옵션을 선택하세요",
        ("기본", "sns", "요약"),
    )

    selected_model = st.selectbox(
        "모델을 선택하세요",
        ("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"),
    )


# 이전 대화를 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 새로운 메시지를 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vecotrstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings,
    )

    retriever = vecotrstore.as_retriever()
    return retriever


def create_chain(retriever, model="gpt-4o"):
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf8")

    llm = ChatOpenAI(model_name=model, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if uploaded_files:
    retriever = embed_file(uploaded_files)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

if clear_btn:
    st.session_state["messages"] = []

print_messages()

USER_INPUT = st.chat_input("질문을 입력하세요")

warning_msg = st.empty()

if USER_INPUT:
    # 여기에 챗GPT API 호출 코드를 추가하여 응답을 받아올 수 있습니다.
    st.chat_message("user").write(USER_INPUT)

    response = chain.stream(USER_INPUT)
    with st.chat_message("assistant"):

        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
    add_message("user", USER_INPUT)
    add_message("assistant", response)
else:
    warning_msg.warning("파일을 업로드 해주세요.")
