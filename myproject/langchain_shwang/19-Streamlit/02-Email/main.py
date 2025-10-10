import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain import hub
from langchain_teddynote import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

st.title("Email 요약 봇💬")

# 이메일 본문으로부터 주요 엔티티 추출
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")

# 이전 대화를 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

# 새로운 메시지를 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def create_email_parsing_chain():
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )
    prompt = prompt.partial(format=output_parser.get_format_instructions())
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    chain = prompt | llm | output_parser

    return chain

def create_report_chain():
    prompt = load_prompt("prompts/email-report.yaml", encoding="utf8")

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain

if clear_btn:
    st.session_state["messages"] = []

print_messages()

USER_INPUT = st.chat_input("메일 내용을 입력하세요")

if USER_INPUT:
    # 여기에 챗GPT API 호출 코드를 추가하여 응답을 받아올 수 있습니다.
    st.chat_message("user").write(USER_INPUT)

    email_chain = create_email_parsing_chain()
    answer = email_chain.invoke({"email_conversation": USER_INPUT})

    print(answer)

    # 2) 보낸 사람의 추가 정보 수집(검색)
    params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}  # 검색 파라미터
    search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"), params=params)
    query = f"{answer.person} {answer.company} {answer.email}"

    search_result = search.run(query)
    search_result = eval(search_result)
    search_result_string = "\n".join(search_result)

    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }
    response = report_chain.invoke({"email_summary": report_chain_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 최종 문자열 저장
    add_message("user", USER_INPUT)
    add_message("assistant", ai_answer)
