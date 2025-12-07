import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote import logging

import bs4
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Webloader")


# Streamlit 제목
st.title("webloader 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

if "chain" not in st.session_state:
    st.session_state["chain"] = None


# 대화 기록 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 메시지 추가
def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


def reload_chain():
    st.session_state["chain"] = create_chain(model_name=st.session_state["model"])


with st.sidebar:
    clear_btn = st.button("대화 초기화")

    selected_model = st.selectbox(
        "LLM 선택",
        ["gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
        key="model",
        on_change=reload_chain,
    )

    url = st.text_input(
        "url을 입력하세요.",
        "https://n.news.naver.com/article/437/0000378416",
        key="url",
        on_change=reload_chain,
    )


# 세션별 대화기록 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# 체인 생성
def create_chain(model_name="gpt-4.1-mini"):
    current_url = st.session_state.get("url")
    loader = WebBaseLoader(
        web_path=(current_url),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                ["div", "span"],
                attrs={
                    "class": [
                        "newsct_article _article_body",
                        "media_end_head_title",
                        "go_trans _article_content",
                    ]
                },
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    clean_splits = []

    for doc in splits:
        if isinstance(doc.page_content, dict):
            # dict → 문자열(JSON)로 강제 변환
            doc.page_content = str(doc.page_content)
        clean_splits.append(doc)

    vectorsotre = FAISS.from_documents(
        documents=clean_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorsotre.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 질문-답변 (Question-Ansering)을 수행하는 친절한 AI 어시스턴트입니다.당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
                검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
                한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

                #Context: 
                {context} 

                #Answer:""",
            ),
            ("human", "#Question:\n{question}"),
        ]
    )

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    rag_chain = (
        {
            "context": lambda x: "\n\n".join(
                [doc.page_content for doc in retriever.invoke(x["question"])]
            ),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print(doc.page_content)
    return rag_chain


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
        response = chain.stream({"question": USER_INPUT})

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
