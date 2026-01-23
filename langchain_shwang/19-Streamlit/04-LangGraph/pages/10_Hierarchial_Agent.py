import streamlit as st
import operator
import os
from typing import List, Union, Tuple, Annotated, TypedDict, Optional, Sequence, Literal
from dotenv import load_dotenv
from langchain_teddynote import logging
from enum import Enum
import matplotlib.pyplot as plt

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_teddynote.tools.tavily import TavilySearch
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import functools

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 필요)
load_dotenv()

logging.langsmith("Hierarchial Agent")

st.set_page_config(page_title="Hierarchial Agent", layout="wide")
st.title("🤖 Hierarchial Agent")

# 멤버 Agent 목록 정의
members = ["retrieve", "Researcher", "Coder"]
# 다음 작업자 선택 옵션 목록 정의
options_for_next = ["FINISH"] + members


# --- 1. 상태 및 모델 정의 ---
class MessageType(Enum):
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"


# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage], operator.add
    ]  # Agent 간 공유하는 메시지 목록
    next: str  # 다음으로 라우팅할 에이전트
    documents: Annotated[List[Document], "The documents retrieved"]


class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved documents."""

    # 문서가 질문에 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RouteResponse(BaseModel):

    next: Literal["FINISH", "retrieve", "Researcher", "Coder"]


# Tavily 검색 도구 정의
tavily_tool = TavilySearch(max_results=5)

# Python 코드를 실행하는 도구 정의
python_repl = PythonREPL()


@st.cache_resource
def get_pdf_retriever(file):
    # 임시 파일 저장 및 로드
    file_path = f"./temp_{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


# Python 코드를 실행하는 도구 정의
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code."""
    try:
        result = python_repl.run(code)
    except BaseException as e:
        return f"Failed to execute code. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    # 1. 그림이 감지되면 파일로 저장 (안전한 전달을 위해)
    if plt.get_fignums():
        chart_file = "chart_output.png"
        try:
            plt.savefig(chart_file)
            plt.close()
            # 2. ★ 약속했던 "특수 태그" 붙이기!
            result_str += "\n\n[FIGURE_GENERATED]"
        except Exception as e:
            result_str += f"\n(Chart save failed: {e})"

    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# 지정한 agent와 name을 사용하여 agent 노드를 생성
def agent_node(state, agent, name):
    # agent 호출
    agent_response = agent.invoke(state)
    # agent의 마지막 메시지를 HumanMessage로 변환하여 반환
    return {
        "messages": [
            HumanMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


# --- 3. 노드 함수 정의 ---


def retrieve(state: AgentState):
    print("\n==== RETRIEVE ====\n")
    query = state["messages"][-1].content
    # st.session_state에 저장된 retriever 사용
    if "pdf_retriever" not in st.session_state:
        return {"documents": []}

    docs = st.session_state["pdf_retriever"].invoke(query)
    return {"documents": docs}


def grade_documents(state: AgentState):
    """검색된 문서 평가 노드"""
    print("\n==== GRADE ====\n")
    question = state["messages"][-1].content
    documents = state["documents"]

    # 1. 문서가 아예 없는 경우 (검색 실패)
    if not documents:
        msg = (
            "PDF 문서에서 관련 정보를 찾을 수 없습니다. "
            "즉시 'Researcher' 에이전트를 호출하여 웹 검색을 수행하세요."
        )
        # Supervisor에게 강력한 힌트(지시)를 보냄
        return {
            "documents": [],
            "messages": [HumanMessage(content=msg, name="grade_documents")],
        }

    # 2. 문서가 있는 경우 평가 진행
    llm = st.session_state["llm_model"]
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    filter_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score == "yes":
            filter_docs.append(doc)

    # 3. 평가 후 관련 문서가 하나도 없는 경우
    if not filter_docs:
        msg_content = (
            "검색된 문서들을 검토했으나 질문과 관련된 내용이 없습니다. "
            "따라서 'Researcher' 에이전트를 호출하여 외부 웹 검색을 수행해야 합니다."
        )
    else:
        msg_content = f"PDF 검색 결과 {len(documents)}건 중 {len(filter_docs)}건의 관련 문서를 찾았습니다. 이 정보를 바탕으로 답변하거나, 필요하다면 Coder에게 시각화를 요청하세요."

    return {
        "documents": filter_docs,
        "messages": [HumanMessage(content=msg_content, name="grade_documents")],
    }


# Supervisor Agent 생성
def supervisor_agent(state):
    system_prompt = (
        "You are a supervisor managing {members}.\n"
        "Your goal is to answer the user's request by routing to the right worker.\n\n"
        "### RULES ###\n"
        "1. **Data First**: If the user wants a chart but data is missing, ALWAYS call 'Researcher' first.\n"
        "2. **No Infinite Loops**: If 'Researcher' just reported an error or failure, DO NOT call them again. Respond with FINISH.\n"
        "3. **Sequence**: 'Researcher' (find data) -> 'Coder' (draw chart) -> FINISH.\n"
        "4. **Completion**: Only respond with FINISH when the chart is displayed or the task is impossible."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Select one of: {options}",
            ),
        ]
    ).partial(options=str(options_for_next), members=", ".join(members))

    llm = st.session_state["llm_model"]
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)


# Research Agent 노드 정의
def research_node(state: AgentState) -> AgentState:
    # Research Agent 생성
    research_agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=[tavily_tool])

    # research node 생성
    research_node = functools.partial(
        agent_node, agent=research_agent, name="Researcher"
    )
    return research_node(state)


# Coder Agent 노드 정의
def coder_node(state: AgentState) -> AgentState:
    code_system_prompt = """
    You are a python coding assistant.
    
    ### CRITICAL RULES FOR PLOTTING ###
    1. **NO plt.show()**: NEVER use `plt.show()`. It will cause a backend error and delete your plot from memory.
    2. **Just Plot**: Just write the code to create the plot (e.g., `plt.plot(...)`, `plt.title(...)`).
    3. **Auto-Save**: The system will automatically detect the active figure and save it as a file. You do NOT need to save it yourself.
    
    ### KOREAN FONT SETTINGS (Must Include) ###
    import platform
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    current_os = platform.system()
    if current_os == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
        fontprop = fm.FontProperties(fname=font_path, size=12)
        plt.rc("font", family=fontprop.get_name())
    elif current_os == "Darwin": # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else: # Linux
        try:
            plt.rcParams["font.family"] = "NanumGothic"
        except:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    ###########################################
    """

    coder_agent = create_react_agent(
        ChatOpenAI(model="gpt-4o"),
        tools=[python_repl_tool],
        prompt=code_system_prompt,
    )

    coder_node_func = functools.partial(agent_node, agent=coder_agent, name="Coder")
    return coder_node_func(state)


def router(state: AgentState):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return "continue"


# --- 4. 그래프 빌드 함수 ---


def build_graph():  # 그래프 생성
    workflow = StateGraph(AgentState)

    # 그래프에 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Coder", coder_node)
    workflow.add_node("Supervisor", supervisor_agent)

    # 1. 일반 멤버들 (Researcher, Coder)은 작업 후 바로 Supervisor로 복귀
    # members 리스트에서 "retrieve"는 제외하고 처리하거나, 직접 지정하는 게 안전합니다.
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Coder", "Supervisor")

    # Supervisor -> retrieve (조건부 엣지로 옴)
    # retrieve -> grade_documents (무조건 이동)
    workflow.add_edge("retrieve", "grade_documents")

    # grade_documents -> Supervisor (평가 끝나면 복귀)
    workflow.add_edge("grade_documents", "Supervisor")

    # 3. 조건부 엣지 설정
    # Supervisor가 선택할 수 있는 옵션들 매핑
    conditional_map = {
        "Researcher": "Researcher",
        "Coder": "Coder",
        "retrieve": "retrieve",  # Supervisor는 'retrieve'를 선택하지만
        "FINISH": END,
    }

    def get_next(state):
        return state["next"]

    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)

    # 시작점
    workflow.add_edge(START, "Supervisor")

    return workflow.compile(checkpointer=MemorySaver())


# --- 5. Streamlit 사이드바 및 초기화 ---

with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드 (선택)", type=["pdf"])

    model_name = st.selectbox(
        "OpenAI 모델 선택", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], index=0
    )

    if st.button("대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 모델 및 도구 설정 (매번 갱신)
st.session_state["llm_model"] = ChatOpenAI(model=model_name, temperature=0)

# 기본 도구: 웹 검색
tools = [TavilySearchResults(max_results=3)]

# PDF가 있으면 도구에 추가
if uploaded_file:
    retriever = get_pdf_retriever(uploaded_file)
    st.session_state["pdf_retriever"] = retriever
    retriever_tool = create_retriever_tool(
        retriever, "pdf_search", "Search for information about the uploaded PDF file."
    )
    tools.append(retriever_tool)
    st.success("PDF 처리 완료! 도구에 추가되었습니다.")

st.session_state["tools"] = tools

# 그래프 생성 (한 번만 혹은 파일 변경 시)
if "graph" not in st.session_state or uploaded_file:
    st.session_state["graph"] = build_graph()

# --- 6. 채팅 인터페이스 ---

# 이전 메시지 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 사용자 메시지 저장 및 출력
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 그래프 실행 설정
    config = {"configurable": {"thread_id": "1"}}

    # 스트리밍 실행 (입력 키는 'input' 이어야 함)
    inputs = {"messages": [HumanMessage(content=prompt)]}

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        events = st.session_state["graph"].stream(
            inputs,
            config={
                "configurable": {"thread_id": "1"},
                "recursion_limit": 20,  # 여기서 제한을 설정합니다 (기본값 25)
            },
        )

        for event in events:
            for node_name, values in event.items():
                if "messages" not in values or not values["messages"]:
                    continue

                latest_message = values["messages"][-1]
                content = latest_message.content

                # --- ★ 사용자님이 원하셨던 "타입 결정 로직" 복원 ---
                message_type = MessageType.TEXT

                # Coder가 "[FIGURE_GENERATED]" 태그를 달고 왔다면? -> 그림 모드!
                if node_name == "Coder" and "[FIGURE_GENERATED]" in content:
                    message_type = MessageType.FIGURE
                    # 태그 텍스트는 보기 싫으니 제거해서 깔끔하게 만듦
                    clean_content = content.replace("[FIGURE_GENERATED]", "")
                else:
                    clean_content = content

                # --- ★ 렌더링 로직 ---
                if node_name == "Coder":
                    with st.expander("💻 Coder (코드 작성 및 실행)", expanded=True):

                        # 1. 텍스트/코드 먼저 출력
                        st.markdown(clean_content)

                        # 2. 그림 타입이면 렌더링
                        if message_type == MessageType.FIGURE:
                            chart_file = "chart_output.png"
                            if os.path.exists(chart_file):
                                # st.pyplot() 대신 st.image()를 쓰지만,
                                # 로직 구조는 사용자님의 의도대로 "태그 감지 시 렌더링"입니다.
                                st.image(chart_file, caption="Generated Chart")
                            else:
                                st.error("⚠️ 그래프 태그는 있는데 파일이 없습니다.")

                    full_response = clean_content

                elif node_name == "Researcher":
                    with st.expander("🕵️ Researcher", expanded=True):
                        st.markdown(content)
                    full_response = content

                elif node_name == "retrieve":
                    with st.expander("🔍 Retrieve", expanded=False):
                        st.markdown(content)

                elif node_name == "grade_documents":
                    with st.expander("⚖️ Grade", expanded=False):
                        st.markdown(content)

                # 메인 메시지 업데이트 (텍스트만)
                if "FINAL ANSWER" in full_response:
                    final_view = full_response.replace("FINAL ANSWER", "").strip()
                    message_placeholder.markdown(final_view)
                else:
                    message_placeholder.markdown(full_response)

    # 스트리밍이 끝난 후, 최종 결과를 세션에 저장
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
