import streamlit as st
import operator
from typing import List, Union, Tuple, Annotated, TypedDict, Optional
from dotenv import load_dotenv
from langchain_teddynote import logging

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 필요)
load_dotenv()

logging.langsmith("PLAN & EXCUTE")

st.set_page_config(page_title="Plan and Execute Agent", layout="wide")
st.title("🤖 Plan and Execute Agent with PDF")

# --- 1. 상태 및 모델 정의 ---

class PlanExecute(TypedDict):
    input: Annotated[str, "User's input"]
    plan: Annotated[List[str], "Current plan"]
    past_steps: Annotated[List[Tuple], operator.add]
    response: Annotated[str, "Final response"]

class Plan(BaseModel):
    """Sorted steps to execute the plan"""
    steps: Annotated[List[str], "Different steps to follow, should be in sorted order"]

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    
    # Union 대신 Optional 필드 두 개로 평탄화(Flatten)합니다.
    response: Optional[str] = Field(
        description="Final response to the user. Use this when you have the answer.", 
        default=None
    )
    plan: Optional[List[str]] = Field(
        description="List of remaining steps to follow. Use this if you need to do more steps.", 
        default=None
    )

# --- 2. 헬퍼 함수 (PDF 처리 및 도구 설정) ---

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

# --- 3. 노드 함수 정의 ---

def plan_step(state: PlanExecute):
    """1. 초기 계획 수립 단계"""
    print("---PLAN STEP---")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
Answer in Korean.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    
    # 모델은 st.session_state에서 가져옴
    model = st.session_state["llm_model"]
    planner = planner_prompt | model.with_structured_output(Plan)
    plan = planner.invoke({"messages": [("user", state["input"])]})
    
    return {"plan": plan.steps}

def execute_step(state: PlanExecute):
    """2. 계획 실행 단계 (LangGraph Prebuilt Agent 사용)"""
    print("---EXECUTE STEP---")
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    
    # 에이전트에게 전달할 구체적인 지시사항
    task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing [step 1. {task}]."""

    # 현재 사용 가능한 도구 및 모델 가져오기
    current_tools = st.session_state["tools"]
    model = st.session_state["llm_model"]

    # 1. LangGraph용 ReAct 에이전트 생성
    # (prompt 대신 state_modifier에 시스템 메시지를 넣습니다)
    agent_app = create_react_agent(
        model, 
        current_tools, 
        prompt="You are a helpful assistant. Answer in Korean."
    )
    
    # 2. Agent 실행
    try:
        # LangGraph 에이전트는 입력으로 {"messages": [...]}를 받습니다.
        result = agent_app.invoke({"messages": [("human", task_formatted)]})
        
        # 3. 결과 파싱 (마지막 메시지가 AI의 최종 답변입니다)
        result_content = result["messages"][-1].content
        
    except Exception as e:
        result_content = f"Error executing step: {str(e)}"

    return {
        "past_steps": [(task, result_content)],
    }

def replan_step(state: PlanExecute):
    """3. 재계획 및 종료 판단 단계"""
    print("---REPLAN STEP---")
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

Answer in Korean."""
    )
    
    model = st.session_state["llm_model"]
    replanner = replanner_prompt | model.with_structured_output(Act)
    
    output = replanner.invoke(state)
    # 1. response 필드가 채워져 있다면 -> 종료 및 답변 반환
    if output.response:
        return {"response": output.response}
    
    # 2. plan 필드가 채워져 있다면 -> 다음 계획 실행
    elif output.plan:
        return {"plan": output.plan}
        
    # 3. 예외 처리 (둘 다 비어있을 경우)
    else:
        return {"response": "죄송합니다. 계획을 수립하는 데 실패했습니다."}

def should_end(state: PlanExecute):
    """조건부 엣지 로직"""
    if "response" in state and state["response"]:
        return "final_report"
    else:
        return "execute"

def generate_final_report(state: PlanExecute):
    """4. 최종 보고서 작성"""
    print("---FINAL REPORT---")
    final_report_prompt = ChatPromptTemplate.from_template(
        """You are given the objective and the previously done steps. Your task is to generate a final report in markdown format.
Final report should be written in professional tone.

Your objective was this:
{input}

Your previously done steps(question and answer pairs):
{past_steps}

Generate a final report in markdown format. Write your response in Korean."""
    )
    
    model = st.session_state["llm_model"]
    final_report = final_report_prompt | model | StrOutputParser()
    
    # past_steps 포맷팅
    past_steps_str = "\n\n".join(
        [f"Question: {step[0]}\nAnswer: {step[1]}" for step in state["past_steps"]]
    )
    
    response = final_report.invoke({"input": state["input"], "past_steps": past_steps_str})
    return {"response": response}

# --- 4. 그래프 빌드 함수 ---

def build_graph():
    workflow = StateGraph(PlanExecute)

    workflow.add_node("planner", plan_step)
    workflow.add_node("execute", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("final_report", generate_final_report)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "execute")
    workflow.add_edge("execute", "replan")
    
    workflow.add_conditional_edges(
        "replan",
        should_end,
        {"execute": "execute", "final_report": "final_report"},
    )
    workflow.add_edge("final_report", END)

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
    inputs = {"input": prompt}
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 이벤트 루프: 실제 그래프의 노드 이름에 맞춰 처리
        events = st.session_state["graph"].stream(inputs, config=config)
        
        for event in events:
            for node_name, values in event.items():
                
                # 1. 계획 수립 단계
                if node_name == "planner":
                    plan_text = "\n".join([f"- {step}" for step in values["plan"]])
                    with st.expander("📅 초기 계획 수립", expanded=True):
                        st.markdown(plan_text)
                        
                # 2. 실행 단계
                elif node_name == "execute":
                    last_step = values["past_steps"][-1] # (Task, Result)
                    with st.expander(f"⚙️ 실행 중: {last_step[0]}", expanded=False):
                        st.write(last_step[1])
                        
                # 3. 재계획 단계
                elif node_name == "replan":
                    if "response" in values:
                        pass # 종료 신호
                    else:
                        new_plan = values.get("plan", [])
                        if new_plan:
                            with st.expander("🔄 계획 수정됨", expanded=False):
                                st.markdown("\n".join([f"- {s}" for s in new_plan]))

                # 4. 최종 보고서
                elif node_name == "final_report":
                    full_response = values["response"]
                    message_placeholder.markdown(full_response)

    # 최종 응답 저장
    st.session_state["messages"].append({"role": "assistant", "content": full_response})