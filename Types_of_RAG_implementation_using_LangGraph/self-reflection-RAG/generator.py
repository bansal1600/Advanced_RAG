from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = prompt | llm | StrOutputParser()

######## Question Re-writer ##########

# LLM 
llm = ChatOpenAI()

# Prompt 
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

question_rewriter_chain = re_write_prompt | llm | StrOutputParser()
