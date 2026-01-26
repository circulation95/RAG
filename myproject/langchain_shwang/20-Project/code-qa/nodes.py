from langchain_core.documents import Document
from chains import (
    create_question_router_chain,
    create_question_rewrite_chain,
    create_retrieval_grader_chain,
    create_groundedness_checker_chain,
    create_answer_grade_chain,
)
from tools import create_web_search_tool

from states import GraphState
from abc import ABC, abstractmethod


### 예시 ###
class BaseNode(ABC):
    def __init__(self, **kwargs):
        self.name = "BaseNode"
        self.verbose = False
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        pass

    def logging(self, method_name, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {method_name}")
            for key, value in kwargs.items():
                print(f"{key}: {value}")

    def __call__(self, state: GraphState):
        return self.execute(state)


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


class WebSearchNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WebSearchNode"
        self.web_search_tool = create_web_search_tool()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        web_results = self.web_search_tool.invoke({"query": question})
        web_results_docs = [
            Document(
                page_content=web_result["content"],
                metadata={"source": web_result["url"]},
            )
            for web_result in web_results
        ]
        return GraphState(documents=web_results_docs)


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
