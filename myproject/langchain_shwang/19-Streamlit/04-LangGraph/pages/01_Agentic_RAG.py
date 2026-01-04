from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
from rag.pdf import PDFRetrievalChain
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_teddynote.models import get_model_name, LLMs
from langchain_core.tools.retriever import create_retriever_tool

load_dotenv()
logging.langsmith("Agent RAG")

st.title("Agentic RAG")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화


# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    TOOL = "tool"  # 도구 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


# 에이전트 상태를 정의하는 타입 딕셔너리, 메시지 시퀀스를 관리하고 추가 동작 정의
class AgentState(TypedDict):
    # add_messages reducer 함수를 사용하여 메시지 시퀀스를 관리
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 데이터 모델 정의
class grade(BaseModel):
    """A binary score for relevance checks"""

    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the question or 'no' if it is not."
    )


def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    pdf = PDFRetrievalChain([file_path]).create_chain()
    return pdf


def grade_documents(state) -> Literal["generate", "rewrite"]:
    # LLM 모델 초기화
    model = ChatOpenAI(temperature=0, model=selected_model, streaming=True)

    # 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(grade)

    # 프롬프트 템플릿 정의
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # llm + tool 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 현재 상태에서 메시지 추출
    messages = state["messages"]

    # 가장 마지막 메시지 추출
    last_message = messages[-1]

    # 원래 질문 추출
    question = messages[0].content

    # 검색된 문서 추출
    retrieved_docs = last_message.content

    # 관련성 평가 실행
    scored_result = chain.invoke({"question": question, "context": retrieved_docs})

    # 관련성 여부 추출
    score = scored_result.binary_score

    # 관련성 여부에 따른 결정
    if score == "yes":
        print("==== [DECISION: DOCS RELEVANT] ====")
        return "generate"

    else:
        print("==== [DECISION: DOCS NOT RELEVANT] ====")
        print(score)
        return "rewrite"


def agent(state):
    # 현재 상태에서 메시지 추출
    messages = state["messages"]

    # LLM 모델 초기화
    model = ChatOpenAI(temperature=0, streaming=True, model=selected_model)

    # retriever tool 바인딩
    model = model.bind_tools(tools)

    # 에이전트 응답 생성
    response = model.invoke(messages)

    # 기존 리스트에 추가되므로 리스트 형태로 반환
    return {"messages": [response]}


def rewrite(state):
    print("==== [QUERY REWRITE] ====")
    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    # 원래 질문 추출
    question = messages[0].content

    # 질문 개선을 위한 프롬프트 구성
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # LLM 모델로 질문 개선
    model = ChatOpenAI(temperature=0, model=selected_model, streaming=True)
    # Query-Transform 체인 실행
    response = model.invoke(msg)

    # 재작성된 질문 반환
    return {"messages": [response]}


def generate(state):
    # 현재 상태에서 메시지 추출
    messages = state["messages"]

    # 원래 질문 추출
    question = messages[0].content

    # 가장 마지막 메시지 추출
    docs = messages[-1].content

    # RAG 프롬프트 템플릿 가져오기
    prompt = hub.pull("teddynote/rag-prompt")

    # LLM 모델 초기화
    llm = ChatOpenAI(model_name=selected_model, temperature=0, streaming=True)

    # RAG 체인 구성
    rag_chain = prompt | llm | StrOutputParser()

    # 답변 생성 실행
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


# 사이드바 설정
with st.sidebar:
    clear_btn = st.button("대화 초기화")  # 대화 내용을 초기화하는 버튼
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드 해주세요.", type=["csv"], accept_multiple_files=True
    )  # CSV 파일 업로드 기능
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0
    )  # OpenAI 모델 선택 옵션
    apply_btn = st.button("데이터 분석 시작")  # 데이터 분석을 시작하는 버튼


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
            st.write(ai_answer)

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


# 메인 로직
if clear_btn:
    st.session_state["messages"] = []  # 대화 내용 초기화

if uploaded_file:
    pdf = embed_file(uploaded_file)
    pdf_chain = pdf.chain
    pdf_retriever = pdf.retriever
    st.session_state["chain"] = pdf_chain

    retriever_tool = create_retriever_tool(
        pdf_retriever,
        "pdf_retriever",
        "Search and return information about SPRI AI Brief PDF file. It contains useful information on recent AI trends. The document is published on Dec 2023.",
        document_prompt=PromptTemplate.from_template(
            "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
        ),
    )
    tools = [retriever_tool]
    st.session_state["tool"] = tools

print_messages()  # 저장된 메시지 출력

user_input = st.chat_input("궁금한 내용을 물어보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리
