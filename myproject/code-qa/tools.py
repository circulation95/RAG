from langchain_teddynote.tools.tavily import TavilySearch


def create_web_search_tool():
    # 웹 검색 도구 생성
    web_search_tool = TavilySearch(max_results=6)
    return web_search_tool
