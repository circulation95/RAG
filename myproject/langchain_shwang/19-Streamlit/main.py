import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
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
    clear = st.button("대화 초기화")

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


def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def create_chain(prompt_type=option):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친철한 ai 어시스턴트입니다."),
            ("user", "#question:\n{question}"),
        ]
    )
    if prompt_type == "sns":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf8")
    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-map-korean", refresh=True)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain


print_messages()
USER_INPUT = st.chat_input("질문을 입력하세요")

if USER_INPUT:
    # 여기에 챗GPT API 호출 코드를 추가하여 응답을 받아올 수 있습니다.
    st.chat_message("user").write(USER_INPUT)
    chain = create_chain()
    response = chain.invoke({"question": USER_INPUT})
    st.chat_message("assistant").write(response)

    add_message("user", USER_INPUT)
    add_message("assistant", response)