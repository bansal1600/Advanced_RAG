"""
Hypothetical Document Embeddings (HyDE): 
ypothetical questions can improve RAG retrieval by creating embeddings for document chunks that represent questions the chunk can answer. These embeddings are often more aligned with a user’s query than those derived from the raw chunk text.

HyDE takes this a step further by preserving the original chunk embeddings while generating hypothetical documents based on the user’s question. This method, illustrated in Figure 2, ensures better semantic similarity between user queries and indexed content, enhancing retrieval accuracy.For example, if you input the prompt with the detailed question,
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#step1 :
#A In case it exists
#B Extract the HTML text from the document
#C Each chunk is a H1 or H2 HTML section
#D Only keep content associated with H2 sections        
#E Loader for one destination
#F Documents of one destination

uk_granular_collection = Chroma(
    collection_name="uk_granular",
    embedding_function=OpenAIEmbeddings(),
)

uk_granular_collection.reset_collection() #A

uk_destinations = [
    "Cornwall", "North_Cornwall"
]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"

uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

headers_to_split_on = [("h1", "Header 1"),("h2", "Header 2")]
html_section_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

def split_docs_into_granular_chunks(docs):
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content #B
        temp_chunks = html_section_splitter.split_text(html_string) #C
        h2_temp_chunks = [chunk for chunk in temp_chunks if "Header 2" in chunk.metadata] #D
        all_chunks.extend(h2_temp_chunks) 

    return all_chunks

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) #E
    docs =  html_loader.load() #F
    
    for doc in docs:
        print(doc.metadata)
        granular_chunks = split_docs_into_granular_chunks(docs)
        uk_granular_collection.add_documents(documents=granular_chunks)

llm = ChatOpenAI(model="gpt-4o-mini")

########### Setting Up the HyDE Chain #################

hyde_prompt_template = """
Write one sentence that could answer the provided question. Do not add anything else.
Question: {question}
Sentence:
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()
user_question = "What are the best beaches in Cornwall?"
hypotetical_document = hyde_chain.invoke(user_question)
###############################step 3###################################################### 
# Combining Everything into a Single RAG Chain
#A The context is returned by the retriver after feeding to it the rewritten query
#B This is the original user question

retriever = uk_granular_collection.as_retriever()

rag_prompt_template = """
Given a question and some context, answer the question.
Only use the provided context to answer the question.
If you do not the answer, just say I do not know. 

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

hyde_rag_chain = (
    {
        "context": {"question": RunnablePassthrough()} | hyde_chain | retriever,#A
        "question": RunnablePassthrough(),#B
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
#A The context is returned by the retriver after feeding to it the hypotetical document
#B This is the original user question

user_question = "What are the best beaches in Cornwall?"

answer = hyde_rag_chain.invoke(user_question)
print(answer)