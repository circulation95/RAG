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
logging.langsmith("Corrective RAG")

st.title("Corrective RAG")

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
    question: Annotated[str, "The question to answer"]
    generation: Annotated[str, "The generation from the LLM"]
    web_search: Annotated[str, "Whether to add search"]
    documents: Annotated[List[str], "The documents retrieved"]

# 검색된 문서의 관련성 여부를 이진 점수로 평가하는 데이터 모델
class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""

    # 문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

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

# 노드 정의
def retrieve(state: State):
    print("\n==== RETRIEVE ====\n")
    query = state["question"]
    # st.session_state에 저장된 retriever 사용
    if "pdf_retriever" not in st.session_state:
        return {"documents": []}
        
    docs = st.session_state["pdf_retriever"].invoke(query)
    return {"documents": docs}

def generate(state: State):
    print("\n==== GENERATE ====\n")
    
    prompt = hub.pull("teddynote/rag-prompt")

    llm = ChatOpenAI(model = selected_model, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()

    question = state["question"]
    documents = state["documents"]

    # RAG를 사용한 답변 생성
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}
    

def grade_documents(state: State):
    print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatOpenAI(model=selected_model, temperature=0)
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
    if relevant_docs:
        web_search = "no"
        filter_docs = documents
    else:
        web_search = "yes"

    return {"documents": documents, "web_search": web_search}

def query_rewrite(state: State):
    print("\n==== [REWRITE QUERY] ====\n")
    question = state["question"]

    llm = ChatOpenAI(model = selected_model, temperature=0)
        
    # Query Rewrite 시스템 프롬프트
    system = """You a question re-writer that converts an input question to a better version that is optimized 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

    # 프롬프트 정의
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    
    # Question Re-writer 체인 초기화
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    better_question = question_rewriter.invoke({"question":question})

    return {"question": better_question}

def web_search(state: State):
    print("\n==== [WEB SEARCH] ====\n")
    question = state["question"]
    documents = state["documents"]
    
    web_search_tool = TavilySearch(max_results=3)

    docs = web_search_tool.invoke({"query":question})
    
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents}

def decide_to_generation(state: State):
    # 평가된 문서를 기반으로 다음 단계 결정
    print("==== [ASSESS GRADED DOCUMENTS] ====")
    web_search = state["web_search"]

    if web_search =="yes":
        # 웹 검색으로 정보 보강이 필요한 경우
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, QUERY REWRITE] ===="
        )
        # 쿼리 재작성 노드로 라우팅
        return "query_rewrite"
    else:
        # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
        print("==== [DECISION: GENERATE] ====")
        return "generate"


def build_graph():# 그래프 상태 초기화
    workflow = StateGraph(State)

    # 노드 정의
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("query_rewrite", query_rewrite)
    workflow.add_node("web_search_node", web_search)

    # 엣지 연결
    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve","grade_documents")
    
    # 조건부 상태 전환 정의
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generation,
        {
            "query_rewrite": "query_rewrite",
            "generate": "generate",
        }
    )

    workflow.add_edge("query_rewrite","web_search_node")
    workflow.add_edge("web_search_node","generate")
    workflow.add_edge("generate", END)

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
    
    # PDF 로직이므로 type을 pdf로 변경
    uploaded_file = st.file_uploader("PDF 파일을 업로드 해주세요.", type=["pdf"])

    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )

    clear_btn = st.button("대화 초기화")

    if st.button("그래프 구조 보기"):
        graph_img = st.session_state["graph"].get_graph().draw_mermaid_png()
        st.image(graph_img)


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
        {"question": query},
        config=config,
    )

# 3. 이벤트 루프: 각 노드(AI, 시뮬레이션 유저)의 출력을 실시간으로 처리for event in events:
    for event in events:
        for node_name, values in event.items():
            content = ""
            st_role = "assistant"
            
            # 노드별로 출력할 내용 결정
            if node_name == "retrieve":
                content = f"📄 문서를 {len(values['documents'])}개 검색했습니다."
                st_role = "tool"
            elif node_name == "grade_documents":
                # web_search 값에 따라 상태 출력
                need_search = values.get("web_search", "no")
                content = f"🔍 문서 평가 완료. (웹 검색 필요: {need_search})"
                st_role = "tool"
            elif node_name == "query_rewrite":
                new_q = values.get("question", "")
                content = f"🔄 질문을 재작성했습니다: {new_q}"
                st_role = "tool"
            elif node_name == "web_search_node":
                content = "🌐 웹 검색을 수행하여 정보를 보강했습니다."
                st_role = "tool"
            elif node_name == "generate":
                # 최종 답변
                content = values["generation"]
                st_role = "assistant"
            
            # 내용이 있으면 출력 및 저장
            if content:
                with st.chat_message(st_role):
                    st.markdown(content)
                add_message(st_role, [MessageType.TEXT, content])

# 메인 로직
if clear_btn:
    st.session_state["messages"] = []
    st.session_state.pop("thread_id", None)  # 새 thread로 시작

if st.session_state["graph"] is None:
    st.session_state["graph"] = build_graph()

if uploaded_file:
    pdf = embed_file(uploaded_file)
    pdf_retriever = pdf.retriever
    pdf_chain = pdf.chain
    st.session_state["pdf_retriever"] = pdf.retriever
    st.session_state["graph"] = build_graph()
    st.success("시스템 준비 완료! 질문을 입력하세요.")

print_messages()
# 사용자 입력 처리 (채팅바)
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    ask(user_input)
