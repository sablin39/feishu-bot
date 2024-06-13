from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain


class DummyAgent:
    def __init__(self, chat_model:ChatOpenAI):
        self.chat_model=chat_model
    def answer_question(self, user_input_question):
        messages = [
                ("system", "You are a financial assistant aiming to analyze provided information for investigation. Based on the user input, provide what the user may need."),
                ("human", user_input_question),
            ]
        result=self.chat_model.invoke(messages)
        return result.content


class QuestionAnsweringSystem:
    def __init__(self, chat_model, vecstore_directory="./userdata/chroma_db_oai"):
        # Set up environment variables for API keys
        
        # Initialize components
        self.chat_model = chat_model
        self.vector_store = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=vecstore_directory)
        self.conversation_memory = ConversationSummaryBufferMemory(llm=self.chat_model, input_key='question', output_key='answer', return_messages=True)
        self.google_search = GoogleSearchAPIWrapper()
        self.web_research_retriever = WebResearchRetriever.from_llm(vectorstore=self.vector_store, llm=self.chat_model, search=self.google_search,trust_env=True)
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.chat_model, retriever=self.web_research_retriever)
    
    def answer_question(self, user_input_question):
        # Query the QA chain with the user input question
        result = self.qa_chain.invoke({"question": user_input_question})
        
        # Return the answer and sources
        return result

# Example usage:
if __name__ == "__main__":
    load_dotenv(dotenv_path="./.env",verbose=True)
    qa_system = QuestionAnsweringSystem(chat_model=ChatOpenAI(model_name="gpt-3.5-turbo", 
                                        temperature=0, 
                                        streaming=True, 
                                        api_key=os.getenv("OPENAI_API_KEY"),
                                        base_url=os.getenv("OPENAI_BASE_URL")),
                                        vecstore_directory="./userdata/chroma_db_oai"
                                        )

    user_input_question =  "Who invented RISC-V" #input("Ask a question: ")
    answer, sources = qa_system.answer_question(user_input_question)
    print("Answer:", answer)
    print("Sources:", sources)