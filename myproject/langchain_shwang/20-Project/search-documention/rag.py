from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.prompts import load_prompt


def create_rag_chain(prompt_name="code-rag-prompt", model_name="gpt-4o"):
    # prompt 설정
    rag_prompt = load_prompt(f"prompts/{prompt_name}.yaml")

    # llm 설정
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 체인(Chain) 생성
    rag_chain = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("context"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
