import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

st.title("나만의 챗GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear = st.button("대화 초기화")

    option = st.selectbox(
        "옵션을 선택하세요",
        ("기본", "sns", "요약"),
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
    chrain = create_chain()
    response = chrain.invoke({"question": USER_INPUT})
    st.chat_message("assistant").write(response)

    add_message("user", USER_INPUT)
    add_message("assistant", response)
