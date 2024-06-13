from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, Tool, create_json_chat_agent, create_openai_tools_agent, create_self_ask_with_search_agent
from langchain.agents import AgentType
import os




google_search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search"
    )
]

class Agent:
    def __init__(self):
        self.agent=initialize_agent(tools = tools, 
                         llm = OpenAI(), 
                         agent=AgentType.SELF_ASK_WITH_SEARCH, 
                         verbose=True)
        
    
    def __call__(self,query:str):
        self.agent.invoke({"input": query})
        return 

