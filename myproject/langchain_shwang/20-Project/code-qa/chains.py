from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "gpt-4o"


class RouteQuery(BaseModel):

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Given a user question, determine if it needs to be retrieved from vectorstore or not. Return 'yes' if it needs to be retrieved from vectorstore, otherwise return 'no'.",
    )


def create_question_router_chain():
    # LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
    system = """You are an expert at routing a user question. 
    The vectorstore contains documents related to RAG(Retrieval Augmented Generation) source code and documentation.
    Return 'yes' if the question is related to the source code or documentation, otherwise return 'no'.
    If you can't determine if the question is related to the source code or documentation, return 'yes'.
    If you don't know the answer, return 'yes'."""

    # Routing 을 위한 프롬프트 템플릿 생성
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # 프롬프트 템플릿과 구조화된 LLM 라우터를 결합하여 질문 라우터 생성
    question_router = route_prompt | structured_llm_router
    return question_router


def create_question_rewrite_chain():
    # LLM 설정
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    # Query Rewrite 시스템 프롬프트
    system = """You a question re-writer that converts an input question to a better version that is optimized for CODE SEARCH(github repository). 
    Look at the input and try to reason about the underlying semantic intent / meaning.

    Base Code Repository: 

    https://github.com/langchain-ai/langgraph

    Output should be in English."""

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
    return question_rewriter


# 문서 평가를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def create_retrieval_grader_chain():
    # LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
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

    # 문서 검색결과 평가기 생성
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader


# 할루시네이션 체크를 위한 데이터 모델 정의
class AnswerGroundedness(BaseModel):
    """Binary score for answer groundedness."""

    binary_score: str = Field(
        description="Answer is grounded in the facts(given context), 'yes' or 'no'"
    )


def create_groundedness_checker_chain():
    # LLM 설정
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_grader = llm.with_structured_output(AnswerGroundedness)

    # 프롬프트 설정
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    # 프롬프트 템플릿 생성
    groundedness_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    # 답변의 환각 여부 평가기 생성
    groundedness_checker = groundedness_prompt | structured_llm_grader
    return groundedness_checker


class GradeAnswer(BaseModel):
    """Binary scoring to evaluate the appropriateness of answers to questions"""

    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )


def create_answer_grade_chain():
    # 함수 호출을 통한 LLM 초기화
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # 프롬프트 설정
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    relevant_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    # 프롬프트 템플릿과 구조화된 LLM 평가기를 결합하여 답변 평가기 생성
    relevant_answer_checker = relevant_answer_prompt | structured_llm_grader
    return relevant_answer_checker
