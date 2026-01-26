from langchain_core.documents import Document
from chains import *
from tools import *
from langgraph.prebuilt import create_react_agent

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from states import *
from abc import ABC, abstractmethod

class BaseNode(ABC):
    def __init__(self, **kwargs):
        self.name = "BaseNode"
        # 모델은 인스턴스 변수로 관리 (재사용성)
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        self.verbose = False
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

    @abstractmethod
    def execute(self, state: GraphState) -> dict:
        """
        LangGraph 노드는 반드시 state 업데이트(dict)를 반환해야 합니다.
        """
        pass

    def create_agent_wrapper(self, agent, name: str):
        """
        Agent 실행 결과를 GraphState 형식에 맞게 변환하는 래퍼 함수
        """
        def run_agent(state):
            result = agent.invoke(state)
            # 마지막 메시지(AI의 답변)를 가져옴
            last_message = result["messages"][-1]
            
            # [수정 포인트] 원본 의도대로 HumanMessage로 변환하여 name 태깅
            # 만약 그냥 AI 답변으로 쓰고 싶다면 AIMessage 그대로 반환해도 됨
            return {
                "messages": [
                    HumanMessage(content=last_message.content, name=name)
                ]
            }
        return run_agent

    def logging(self, method_name, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {method_name}")
            for key, value in kwargs.items():
                print(f"  - {key}: {value}")

    def __call__(self, state: GraphState):
        self.logging("Execution Started")
        result = self.execute(state)
        self.logging("Execution Finished")
        return result


class WebSearchNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WebSearcher"
        
        self.tools = [create_web_search_tool()]
        self.agent_executor = create_react_agent(self.llm, tools=self.tools)
        
        self.runnable_agent = self.create_agent_wrapper(self.agent_executor, self.name)

    def execute(self, state: ResearchState) -> dict:
        self.logging("Searching...", query=state.get("messages", [])[-1].content[:50])
        
        return self.runnable_agent(state)

class WebScrapingNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WebScraper"
        
        self.tools = [scrape_webpages()]
        self.agent_executor = create_react_agent(self.llm, tools=self.tools)
        
        self.runnable_agent = self.create_agent_wrapper(self.agent_executor, self.name)

    def execute(self, state: ResearchState) -> dict:
        self.logging("Scraping...", query=state.get("messages", [])[-1].content[:50])
        
        return self.runnable_agent(state)
    

class RouteQuestionNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RouteQuestionNode"
        self.router_chain = create_question_router_chain()

    def execute(self, state: GraphState) -> str:
        question = state["question"]
        evaluation = self.router_chain.invoke({"question": question})

        if evaluation.binary_score == "yes":
            return "query_expansion"
        else:
            return "general_answer"


class QueryRewriteNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "QueryRewriteNode"
        self.rewriter_chain = create_question_rewrite_chain()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        better_question = self.rewriter_chain.invoke({"question": question})
        return GraphState(question=better_question)


class RetrieveNode(BaseNode):
    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.name = "RetrieveNode"
        self.retriever = retriever

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = self.retriever.invoke(question)
        return GraphState(documents=documents)


class GeneralAnswerNode(BaseNode):
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.name = "GeneralAnswerNode"
        self.llm = llm

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        answer = self.llm.invoke(question)
        return GraphState(generation=answer.content)


class RagAnswerNode(BaseNode):
    def __init__(self, rag_chain, **kwargs):
        super().__init__(**kwargs)
        self.name = "RagAnswerNode"
        self.rag_chain = rag_chain

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        answer = self.rag_chain.invoke({"context": documents, "question": question})
        return GraphState(generation=answer)


class FilteringDocumentsNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "FilteringDocumentsNode"
        self.retrieval_grader = create_retrieval_grader_chain()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            if score.binary_score == "yes":
                filtered_docs.append(d)

        return GraphState(documents=filtered_docs)



class AnswerGroundednessCheckNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AnswerGroundednessCheckNode"
        self.groundedness_checker = create_groundedness_checker_chain()
        self.relevant_answer_checker = create_answer_grade_chain()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.groundedness_checker.invoke(
            {"documents": documents, "generation": generation}
        )

        if score.binary_score == "yes":
            score = self.relevant_answer_checker.invoke(
                {"question": question, "generation": generation}
            )
            if score.binary_score == "yes":
                return "relevant"
            else:
                return "not relevant"
        else:
            return "not grounded"


# 추가 정보 검색 필요성 여부 평가 노드
def decide_to_web_search_node(state):
    # 문서 검색 결과 가져오기
    filtered_docs = state["documents"]

    if len(filtered_docs) < 2:
        return "web_search"
    else:
        return "rag_answer"

def get_next_node(x):
    return x["next"]