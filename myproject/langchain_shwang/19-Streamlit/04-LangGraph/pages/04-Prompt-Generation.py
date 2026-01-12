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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from typing import Literal
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_teddynote.models import get_model_name, LLMs
from langchain_core.tools.retriever import create_retriever_tool

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid
from langchain_core.documents import Document
from langchain_teddynote.tools.tavily import TavilySearch

load_dotenv()
logging.langsmith("Agent Simulation RAG")

st.title("Agent Simulation RAG")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "graph" not in st.session_state:
    st.session_state["graph"] = None


# 상수 정의
class MessageRole:
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"


class MessageType:
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    DATAFRAME = "dataframe"


# State 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]


# LLM에 대한 프롬프트 지침을 정의하는 데이터 모델
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    # 프롬프트의 목표
    objective: str
    # 프롬프트 템플릿에 전달될 변수 목록
    variables: List[str]
    # 출력에서 피해야 할 제약 조건 목록
    constraints: List[str]
    # 출력이 반드시 따라야 할 요구 사항 목록
    requirements: List[str]


# 함수 정의 (기존 유지)
def format_docs(docs):
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )


def embed_file(file):
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    pdf = PDFRetrievalChain([file_path]).create_chain()
    return pdf


# 사용자 메시지 목록을 받아 시스템 메시지와 결합하여 반환
def get_messages_info(messages):
    # 사용자 요구사항 수집을 위한 시스템 메시지 템플릿
    template = """Your job is to get information from a user about what type of prompt template they want to create.

    You should get the following information from them:

    - What the objective of the prompt is
    - What variables will be passed into the prompt template
    - Any constraints for what the output should NOT do
    - Any requirements that the output MUST adhere to

    If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

    After you are able to discern all the information, call the relevant tool.

    [IMPORTANT] Your conversation should be in Korean. Your generated prompt should be in English."""
    # 사용자 요구사항 수집을 위한 시스템 메시지와 기존 메시지 결합
    return [SystemMessage(content=template)] + messages


# 상태 정보를 기반으로 메시지 체인을 생성하고 LLM 호출
def info_chain(state):
    llm = ChatOpenAI(temperature=0, model=selected_model)
    # PromptInstructions 구조체를 바인딩
    llm_with_tool = llm.bind_tools([PromptInstructions])
    # 상태에서 메시지 정보를 가져와 시스템 메시지와 결합
    messages = get_messages_info(state["messages"])
    # LLM을 호출하여 응답 생성
    response = llm_with_tool.invoke(messages)
    # 생성된 응답을 메시지 목록으로 반환
    return {"messages": [response]}


# 프롬프트 생성을 위한 메시지 가져오기 함수
# 도구 호출 이후의 메시지만 가져옴
def get_prompt_messages(messages: list):

    # 프롬프트를 생성하는 메타 프롬프트 정의(OpenAI 메타 프롬프트 엔지니어링 가이드 참고)
    META_PROMPT = """Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

    # Guidelines

    - Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
    - Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
    - Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
        - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
        - Conclusion, classifications, or results should ALWAYS appear last.
    - Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
    - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
    - Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
    - Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
    - Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
    - Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
    - Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
        - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
        - JSON should never be wrapped in code blocks (```) unless explicitly requested.

    The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

    [Concise instruction describing the task - this should be the first line in the prompt, no section header]

    [Additional details as needed.]

    [Optional sections with headings or bullet points for detailed steps.]

    # Steps [optional]

    [optional: a detailed breakdown of the steps necessary to accomplish the task]

    # Output Format

    [Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

    [User given variables should be wrapped in {{brackets}}]

    <Question>
    {{question}}
    </Question>

    <Answer>
    {{answer}}
    </Answer>

    # Examples [optional]

    [Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
    [If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

    # Notes [optional]

    [optional: edge cases, details, and an area to call or repeat out specific important considerations]

    # Based on the following requirements, write a good prompt template:

    {reqs}
    """
    # 도구 호출 정보를 저장할 변수 초기화
    tool_call = None
    # 도구 호출 이후의 메시지를 저장할 리스트 초기화
    other_msgs = []
    # 메시지 목록을 순회하며 도구 호출 및 기타 메시지 처리
    for m in messages:
        # AI 메시지 중 도구 호출이 있는 경우 도구 호출 정보 저장
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        # ToolMessage는 건너뜀
        elif isinstance(m, ToolMessage):
            continue
        # 도구 호출 이후의 메시지를 리스트에 추가
        elif tool_call is not None:
            other_msgs.append(m)
    # 시스템 메시지와 도구 호출 이후의 메시지를 결합하여 반환
    return [SystemMessage(content=META_PROMPT.format(reqs=tool_call))] + other_msgs


# 프롬프트 생성 체인 함수 정의
def prompt_gen_chain(state):
    llm = ChatOpenAI(temperature=0, model=selected_model)
    # 상태에서 프롬프트 메시지를 가져옴
    messages = get_prompt_messages(state["messages"])
    # LLM을 호출하여 응답 생성
    response = llm.invoke(messages)
    # 생성된 응답을 메시지 목록으로 반환
    return {"messages": [response]}


# 상태 결정 함수 정의
# 상태에서 메시지 목록을 가져옴
def get_state(state):
    messages = state["messages"]
    # 마지막 메시지가 AIMessage이고 도구 호출이 있는 경우
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        # 도구 메시지를 추가해야 하는 상태 반환
        return "add_tool_message"
    # 마지막 메시지가 HumanMessage가 아닌 경우
    elif not isinstance(messages[-1], HumanMessage):
        # 대화 종료 상태 반환
        return END
    # 기본적으로 정보 수집 상태 반환
    return "info"


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("info", info_chain)
    workflow.add_node("prompt", prompt_gen_chain)

    # 도구 메시지 추가 상태 노드 정의
    @workflow.add_node
    def add_tool_message(state: State):
        return {
            "messages": [
                ToolMessage(
                    content="Prompt generated!",
                    tool_call_id=state["messages"][-1].tool_calls[0][
                        "id"
                    ],  # 상태에서 도구 호출 ID를 가져와 메시지에 추가
                )
            ]
        }

    # 조건부 상태 전환 정의
    workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])

    # 엣지 정의
    workflow.add_edge("add_tool_message", "prompt")
    workflow.add_edge("prompt", END)
    workflow.add_edge(START, "info")

    return workflow.compile(checkpointer=MemorySaver())


def print_messages():
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    if len(content) == 2:
                        message_type, message_content = content
                    elif len(content) == 1:
                        message_type = MessageType.TEXT
                        message_content = content[0]
                    else:
                        continue

                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)
                    elif message_type == MessageType.CODE:
                        st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)
                elif isinstance(content, str):
                    st.markdown(content)


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])


# --- 사이드바 설정 ---
with st.sidebar:

    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )

    clear_btn = st.button("대화 초기화")


# 질문 처리 함수 (수정됨: name, instructions 전달)
# 질문 처리 함수 (수정됨: stream 사용)
def ask(query):
    # 1. 초기 트리거 메시지(사용자 입력) 저장 및 출력
    add_message(MessageRole.USER, [MessageType.TEXT, query])
    with st.chat_message("user"):
        st.write(query)

    graph = st.session_state["graph"]

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = random_uuid()

    config = RunnableConfig(
        recursion_limit=20,  # 대화가 길어질 수 있으니 제한을 조금 늘림
        configurable={"thread_id": st.session_state["thread_id"]},
    )

    # 2. Graph 스트리밍 실행
    # graph.stream을 쓰면 노드 하나가 끝날 때마다 event를 반환합니다.
    events = graph.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    # 3. 이벤트 루프: 각 노드(AI, 시뮬레이션 유저)의 출력을 실시간으로 처리
    for event in events:
        for node_name, values in event.items():
            if "messages" not in values or not values["messages"]:
                continue

            last_message = values["messages"][-1]

            # 1) tuple 형태 (role, content)인 경우
            if isinstance(last_message, tuple):
                role, content = last_message
                st_role = "user" if role == "user" else "assistant"

            else:
                # 2) LangChain 메시지 객체인 경우
                if isinstance(last_message, HumanMessage):
                    st_role = "user"
                    content = last_message.content

                elif isinstance(last_message, ToolMessage):
                    st_role = "tool"
                    content = last_message.content

                elif isinstance(last_message, AIMessage):
                    st_role = "assistant"
                    content = last_message.content or ""

                    # tool_calls만 있고 content가 비는 케이스 보정(선택)
                    if not content.strip() and getattr(
                        last_message, "tool_calls", None
                    ):
                        content = "요구사항을 정리했습니다. 다음 단계로 진행할게요."

                    # 진짜로 아무것도 없으면 출력 스킵
                    if not content.strip():
                        continue

                else:
                    # 기타 메시지 타입 fallback
                    st_role = "assistant"
                    content = getattr(last_message, "content", "") or ""
                    if not content.strip():
                        continue

            # 3) 출력
            with st.chat_message(st_role):
                st.markdown(content)

            # 4) 저장
            add_message(st_role, [MessageType.TEXT, content])


# 메인 로직
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["graph"] = build_graph()  # 새 checkpointer
    st.session_state.pop("thread_id", None)  # 새 thread로 시작

if st.session_state["graph"] is None:
    st.session_state["graph"] = build_graph()

print_messages()
# 사용자 입력 처리 (채팅바)
user_input = st.chat_input("시뮬레이션을 시작하려면 메시지를 입력하세요!")

if user_input:
    ask(user_input)
