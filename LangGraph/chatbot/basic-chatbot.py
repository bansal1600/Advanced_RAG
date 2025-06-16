from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatOpenAI()

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)
graph.set_entry_point("chatbot")
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)

app = graph.compile()

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        print(result)