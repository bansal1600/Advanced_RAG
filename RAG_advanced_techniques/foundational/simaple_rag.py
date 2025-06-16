from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Load environment variables from .env file
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

file_path = 'data/customers-100.csv'

#load the csv data
loader = CSVLoader(file_path=file_path)
data = loader.load_and_split()

#  embeddings in FAISS vector store using from_texts
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vector_db = FAISS.from_documents(data, embedding_model)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# ✨ ADD a small function to fix formatting
def format_docs_for_prompt(docs, question):
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context, "question": question}

# --- Modified flow ---
def rag_chain(question):
    # 1. Retrieve docs
    docs = retriever.invoke(question)
    
    # 2. Format the context for prompt
    prompt_input = format_docs_for_prompt(docs, question)
    
    # 3. Format the messages
    formatted_prompt = prompt.invoke(prompt_input)
    
    # 4. Send to LLM
    answer = llm.invoke(formatted_prompt)
    
    return answer.content

# --- Call like this ---
response = rag_chain("which company does Sheryl Baxter work for?")
print(response)