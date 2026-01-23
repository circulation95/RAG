import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote import logging

from dotenv import load_dotenv
import os

# .env 로드
load_dotenv()
logging.langsmith("[Project] Multi Turn 챗봇")

# 캐시 디렉토리 생성
for folder in [".cache", ".cache/files", ".cache/embeddings"]:
    os.makedirs(folder, exist_ok=True)

# Streamlit 제목
st.title("대화내용을 기억하는 챗봇 💬")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0)
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")


# 대화 기록 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 메시지 추가
def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 세션별 대화기록 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# 체인 생성
def create_chain(model_name="gpt-4.1-mini"):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),
        ]
    )

    llm = ChatOpenAI(model_name=model_name)
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history


# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["chain"] = create_chain(model_name=selected_model)

# 이전 대화 출력
print_messages()

# 체인이 없으면 새로 생성
if st.session_state["chain"] is None:
    st.session_state["chain"] = create_chain(model_name=selected_model)

# 사용자 입력
USER_INPUT = st.chat_input("질문을 입력하세요.")
warning_msg = st.empty()

# 입력이 들어오면
if USER_INPUT:
    chain = st.session_state["chain"]
    if chain is not None:
        response = chain.stream(
            {"question": USER_INPUT},
            config={"configurable": {"session_id": session_id}},
        )

        st.chat_message("user").write(USER_INPUT)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            add_message("user", USER_INPUT)
            add_message("assistant", ai_answer)
    else:
        warning_msg.error("Chain이 초기화되지 않았습니다.")
