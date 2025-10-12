import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os

load_dotenv()

logging.langsmith("[Project] 이미지 인식")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("이미지 인식 기반 챗봇 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    uploaded_files = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"],
    )

    selected_model = st.selectbox(
        "LLM 선택하세요",
        ["gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
    )

    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.\n당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.",
        height=200,
    )


# 이전 대화를 출력
def print_messages():
    for msg in st.session_state["messages"]:
        main_tab2.chat_message(msg.role).write(msg.content)


# 새로운 메시지를 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


def generate_answer(
    image_filepath, system_prompt, user_prompt, model_name="gpt-4.1-mini"
):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model=model_name,
    )

    multimodal = MultiModal(llm, system_prompt, user_prompt)
    answer = multimodal.stream(image_filepath)
    return answer


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

print_messages()

USER_INPUT = st.chat_input("질문을 입력하세요")

warning_msg = main_tab2.empty()

# 파일이 업로드 되었을 때
if uploaded_files:
    image_filepath = process_imagefile(uploaded_files)
    main_tab1.image(image_filepath)

if USER_INPUT:

    if uploaded_files:
        image_filepath = process_imagefile(uploaded_files)
        response = generate_answer(
            image_filepath, system_prompt, USER_INPUT, model_name=selected_model
        )
        main_tab2.chat_message("user").write(USER_INPUT)

        with main_tab2.chat_message("assistant"):

            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        add_message("user", USER_INPUT)
        add_message("assistant", ai_answer)
    else:
        warning_msg.warning("⚠️ 이미지를 업로드 해주세요!")
