import streamlit as st
import operator
import os
from typing import List, Union, Tuple, Annotated, TypedDict, Optional, Sequence
from dotenv import load_dotenv
from langchain_teddynote import logging

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
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

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 필요)
load_dotenv()

logging.langsmith("Multi Agent")

st.set_page_config(page_title="Multi Agent", layout="wide")
st.title("🤖 Multi Agent with PDF")

# --- 1. 상태 및 모델 정의 ---

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage], operator.add
    ]  # Agent 간 공유하는 메시지 목록
    sender: Annotated[str, "The sender of the last message"]  # 마지막 메시지의 발신자
    documents: Annotated[List[Document], "The documents retrieved"]

class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved documents."""

    # 문서가 질문에 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

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
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        # 주어진 코드를 Python REPL에서 실행하고 결과 반환
        result = python_repl.run(code)
    except BaseException as e:
        return f"Failed to execute code. Error: {repr(e)}"
    # 실행 성공 시 결과와 함께 성공 메시지 반환
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

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
    print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
    question = state["messages"][-1].content
    documents = state["documents"]
    
    llm = st.session_state["llm_model"]
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # 시스템 프롬프트 정의
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}") 
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    filter_docs = []
    relevant_docs = 0

    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score

        if grade == "yes":
            print("==== [GRADE: DOCUMENT RELEVANT] ====")
            filter_docs.append(doc)
            relevant_docs += 1
        else:
            print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
            continue
    return {"documents": filter_docs}

# Research Agent 노드 정의
def research_node(state: AgentState) -> AgentState:
    
    llm = st.session_state["llm_model"]

    # Research Agent 생성
    research_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        prompt=make_system_prompt(
            "You can only do research. You are working with a chart generator colleague."
        ),
    )

    result = research_agent.invoke(state)

    # 마지막 메시지를 HumanMessage 로 변환
    last_message = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return {
        # Research Agent 의 메시지 목록 반환
        "messages": [last_message],
    }

def chart_node(state: AgentState) -> AgentState:

    llm = st.session_state["llm_model"]

    documents = state.get("documents", [])
    context_str = "\n\n".join([doc.page_content for doc in documents])
    chart_generator_system_prompt = f"""
    You can only generate charts. You are working with a researcher colleague.
    Be sure to use the following font code in your code when generating charts.

    IMPORTANT: 
    1. Do NOT use `plt.show()`. It will not work.
    2. Instead, save the chart as an image file named 'chart_output.png' using `plt.savefig('chart_output.png')`.
    3. After saving, print "Chart saved to chart_output.png".

    Here is the data you retrieved from the PDF:
    -----
    {context_str}
    -----
    """
    
    # Chart Generator Agent 생성
    chart_agent = create_react_agent(
        llm,
        [python_repl_tool],
        prompt=make_system_prompt(chart_generator_system_prompt),
    )

    result = chart_agent.invoke(state)

    # 마지막 메시지를 HumanMessage 로 변환
    last_message = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return {
        # share internal message history of chart agent with other agents
        "messages": [last_message],
    }

def decide_to_generation(state: AgentState):
    # 평가된 문서를 기반으로 다음 단계 결정
    print("==== [ASSESS GRADED DOCUMENTS] ====")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 웹 검색으로 정보 보강이 필요한 경우
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, RESEARCHER] ===="
        )
        # 쿼리 재작성 노드로 라우팅
        return "researcher"
    else:
        # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
        print("==== [DECISION: GENERATE] ====")
        return "chart_generator"
    
def router(state: AgentState):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return "continue"

# --- 4. 그래프 빌드 함수 ---

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generation,
        {"researcher": "researcher", "chart_generator": "chart_generator"},
    )

    workflow.add_conditional_edges(
        "researcher",
        router,
        {"continue": "chart_generator", END: END},
    )
    workflow.add_conditional_edges(
        "chart_generator",
        router,
        {"continue": "researcher", END: END},
    )

    workflow.add_edge(START, "retrieve")

    return workflow.compile(checkpointer=MemorySaver())

# --- 5. Streamlit 사이드바 및 초기화 ---

with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드 (선택)", type=["pdf"])
    
    model_name = st.selectbox(
        "OpenAI 모델 선택",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
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
        retriever,
        "pdf_search",
        "Search for information about the uploaded PDF file."
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
        
        # 이벤트 루프: 실제 그래프의 노드 이름에 맞춰 처리
        events = st.session_state["graph"].stream(inputs, config=config)
        
        for event in events:
            for node_name, values in event.items():
                # 메시지 목록에서 가장 최신 메시지 가져오기
                if "messages" in values:
                    latest_message = values["messages"][-1]
                    content = latest_message.content
                else:
                    continue # 메시지가 없는 경우 스킵
                
                # 1. Researcher 에이전트의 응답 처리
                if node_name == "researcher":
                    with st.expander("🕵️ Researcher (자료 조사 중...)", expanded=False):
                        st.markdown(content)
                    # 진행 상황을 잠시 보여주기 위해 full_response 업데이트 (선택 사항)
                    full_response = content 
                        
                # 2. Chart Generator 에이전트의 응답 처리
                elif node_name == "chart_generator":
                    with st.expander("📊 Chart Generator (차트 생성 중...)", expanded=True):
                        st.markdown(content)
                        
                        # [추가] 차트 이미지가 생성되었는지 확인하고 출력
                        chart_file = "chart_output.png"
                        if os.path.exists(chart_file):
                            # 캐시 문제 방지를 위해 이미지를 열어서 바로 표시
                            st.image(chart_file, caption="Generated Chart")
                            
                    full_response = content

                # 3. 기타 노드 혹은 종료 조건 처리 (필요시 추가)
                
                # "FINAL ANSWER"가 포함되어 있으면 최종 답변으로 간주하고 정제
                if "FINAL ANSWER" in content:
                    full_response = content.replace("FINAL ANSWER", "").strip()

                # 실시간으로 메인 메시지 업데이트 (현재 단계의 결과물을 계속 보여줌)
                message_placeholder.markdown(full_response)

    # 최종 응답 저장
    st.session_state["messages"].append({"role": "assistant", "content": full_response})