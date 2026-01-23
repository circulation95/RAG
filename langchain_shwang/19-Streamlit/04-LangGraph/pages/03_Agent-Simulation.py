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
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
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
    name: str
    instructions: str
    messages: Annotated[list, add_messages]

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

def call_chatbot(messages: List[BaseMessage]) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional customer support agent. "
                "You can handle a wide range of user inquiries across different domains. "
                "Your goal is to clearly understand the user's intent, ask clarifying questions if needed, "
                "and provide accurate, polite, and helpful responses. "
                "Always respond in Korean."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = ChatOpenAI(model=selected_model, temperature=0.6)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"messages": messages})

def create_scenario(name: str, instructions: str):
    system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

Your name is {name}.

# Instructions:
{instructions}

[IMPORTANT] 
- When you are finished with the conversation, respond with a single word 'FINISHED'
- You must speak in Korean."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(name=name, instructions=instructions)
    return prompt

def _swap_roles(messages):
    print("==== [SWAP ROLES] ====")
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages

def ai_assistant_node(state: State):
    print("==== [AI ASSISTANT] ====")
    ai_response = call_chatbot(state["messages"])
    return {"messages": [("assistant", ai_response)]}

def simulated_user_node(state: State):
    print("==== [SIMULATED USER] ====")
    name = state["name"]
    instructions = state["instructions"]
    llm = ChatOpenAI(model=selected_model, temperature=0.6)
    simulated_user = create_scenario(name, instructions) | llm | StrOutputParser()
    new_messages = _swap_roles(state["messages"])
    response = simulated_user.invoke({"messages": new_messages})
    return {"messages": [("user", response)]}

def should_continue(state: State):
    print("==== [SHOULD CONTINUE] ====")
    if len(state["messages"]) > 6:
        return "end"
    elif state["messages"][-1].content == "FINISHED":
        print("==== [FINISH] ====")
        return "end"
    else:  
        print("==== [CONTINUE] ====")
        return "continue"

def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("simulated_user", simulated_user_node)
    workflow.add_node("ai_assistant", ai_assistant_node)

    workflow.add_edge("ai_assistant", "simulated_user")

    workflow.add_conditional_edges(
        "simulated_user",
        should_continue,
        {
            "end": END,
            "continue": "ai_assistant",
        },
    )

    workflow.set_entry_point("ai_assistant")

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
    st.header("시뮬레이션 설정") # 헤더 추가
    
    # [수정] 이름과 지시사항 입력 필드 추가
    sim_name = st.text_input("시뮬레이션 사용자 이름", placeholder="예: 김철수")
    sim_instructions = st.text_area("상황 및 지시사항", placeholder="예: 항공권 날짜를 다음주로 변경하고 싶음. 환불 수수료에 대해 불만이 있음.")
    
    st.divider() # 구분선
    
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )
    
    clear_btn = st.button("대화 초기화")
    
    execute_btn = st.button("시뮬레이션 실행")


# 질문 처리 함수 (수정됨: name, instructions 전달)
# 질문 처리 함수 (수정됨: stream 사용)
def ask(query, name, instructions):
    # 1. 초기 트리거 메시지(사용자 입력) 저장 및 출력
    add_message(MessageRole.USER, [MessageType.TEXT, query])
    with st.chat_message("user"):
        st.write(query)

    graph = st.session_state["graph"]

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = random_uuid()

    config = RunnableConfig(
        recursion_limit=20, # 대화가 길어질 수 있으니 제한을 조금 늘림
        configurable={"thread_id": st.session_state["thread_id"]},
    )

    # 2. Graph 스트리밍 실행
    # graph.stream을 쓰면 노드 하나가 끝날 때마다 event를 반환합니다.
    events = graph.stream(
        {
            "messages": [HumanMessage(content=query)], 
            "name": name,
            "instructions": instructions
        }, 
        config=config
    )

    # 3. 이벤트 루프: 각 노드(AI, 시뮬레이션 유저)의 출력을 실시간으로 처리
    for event in events:
        for node_name, values in event.items():
            # values["messages"]에는 해당 노드가 반환한 메시지 리스트가 들어있습니다.
            # 예: [('assistant', '안녕하세요...')] 또는 [('user', '환불해주세요...')]
            if "messages" in values:
                last_message = values["messages"][-1] 
                
                # 튜플 형태로 저장된 경우 (role, content) 분리
                if isinstance(last_message, tuple):
                    role, content = last_message
                else:
                    # LangChain Message 객체인 경우 (BaseMessage)
                    role = "user" if isinstance(last_message, HumanMessage) else "assistant"
                    content = last_message.content

                # 4. 역할에 맞게 화면에 즉시 출력
                # role이 'user'면 시뮬레이션 고객, 'assistant'면 상담원
                st_role = "user" if role == "user" else "assistant"
                
                with st.chat_message(st_role):
                    st.markdown(content)
                
                # 5. 세션 스테이트에 저장 (새로고침 시 유지용)
                add_message(st_role, [MessageType.TEXT, content])

# 메인 로직
if clear_btn:
    st.session_state["messages"] = []

if st.session_state["graph"] is None:
    st.session_state["graph"] = build_graph()

print_messages()
# 사용자 입력 처리 (채팅바)
user_input = st.chat_input("시뮬레이션을 시작하려면 메시지를 입력하세요!")

# [추가] 1. '시뮬레이션 실행' 버튼 클릭 시 처리 로직
if execute_btn:
    # 이름과 지시사항이 입력되었는지 확인
    if not sim_name.strip() or not sim_instructions.strip():
        st.warning("⚠️ 사이드바에서 '시뮬레이션 사용자 이름'과 '상황 및 지시사항'을 모두 입력해주세요!")
    else:
        # 강제로 "안녕하세요" 메시지를 보내 시뮬레이션 시작
        ask("안녕하세요", sim_name, sim_instructions)
        # 실행 후 UI 갱신을 위해 rerun (선택사항, ask 함수가 화면에 그리므로 없어도 동작함)
        st.rerun()

# [기존] 2. 사용자가 직접 채팅창에 입력했을 때 처리 로직
if user_input:
    if not sim_name.strip() or not sim_instructions.strip():
        st.warning("⚠️ 사이드바에서 '시뮬레이션 사용자 이름'과 '상황 및 지시사항'을 모두 입력해주세요!")
    else:
        ask(user_input, sim_name, sim_instructions)