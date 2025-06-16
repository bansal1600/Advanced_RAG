from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

genaration_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a twitter techie influencer, you write tweets about tech."
         "generate a tweet for user's request"
         "if user provides critique, respond with a revised tweet"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a viral tweet influencer grding a tweet. generate critique and recommendationfor user's tweet"
         "always provide a detailed recommendation, including requests for length, virality, and engagement"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generation_chain = genaration_prompt | llm
reflection_chain = reflection_prompt | llm