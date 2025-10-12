import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_teddynote import logging
from retriever import create_retriever
from dotenv import load_dotenv
import os

load_dotenv()

logging.langsmith("[Project] Local RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Local 모델 기반 RAG 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    uploaded_files = st.file_uploader(
        "파일 업로드",
        type=["pdf"],
    )

    selected_model = st.selectbox(
        "모델을 선택하세요",
        ["xionic", "ollama"],
        index=0,
    )


# 이전 대화를 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 새로운 메시지를 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


def create_chain(retriever, model_name="xionic"):
    if model_name == "xionic":
        prompt = load_prompt("prompts/pdf-rag-xionic.yaml", encoding="utf8")

        llm = ChatOpenAI(
            model_name="llama-3.1-xionic-ko-70b",
            base_url="http://sionic.chat:8001/v1",
            api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
        )
    elif model_name == "ollama":
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf8")

        llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)

    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일이 업로드 되었을 때
if uploaded_files:
    # 파일 업로드 후 retriever 생성
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
    chain = st.session_state["chain"]
    if chain is not None:
        st.chat_message("user").write(USER_INPUT)

        response = chain.stream(USER_INPUT)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 최종 문자열 저장
        add_message("user", USER_INPUT)
        add_message("assistant", ai_answer)
    else:
        warning_msg.warning("파일을 업로드 해주세요.")
