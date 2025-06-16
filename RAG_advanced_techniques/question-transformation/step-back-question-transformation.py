"""
Step-Back Question Workflow: 
-> The LLM application first sends the detailed question (Q_D) to the vector store to retrieve a detailed context (C_D). 
-> It then prompts the LLM to generate a more abstract question (Q_A) based on Q_D, which is also executed on the vector store to obtain an abstract context (C_A). 
-> Finally, the LLM application combines Q_D, C_D, and C_A into a single prompt, enabling the LLM to synthesize a comprehensive answer.

For example, if you input the prompt with the detailed question,
detailed_question: “Can you give me some tips for a trip to Brighton?” 

the more abstract (step-back) question might look like:
Step-back question: “What should I know before visiting a popular coastal town?”

This broader question helps retrieve more general information, which, combined with the detailed context, allows the LLM to produce a well-rounded answer.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
import getpass
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough, RunnableMap

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

########### Setting Up the MultiQuery Rewriter Chain #################

detailed_question_rewriter_prompt_template = """
Generate search query for the Chroma DB vector store from a user question, allowing for a more accurate response through semantic search.
Just return the revised Chroma DB query, with quotes around it. 

User question: {user_question}
Revised Chroma DB query:
"""

detailed_question_rewriter_prompt_template = ChatPromptTemplate.from_template(detailed_question_rewriter_prompt_template)
detailed_question_rewriter_chain = detailed_question_rewriter_prompt_template | llm | StrOutputParser()

########### Setting Up the abstract question Chain #################

abstarct_question_rewriter_prompt_template = """
Generate a less specific question (aka Step-back question) for the following detailed question, so that a wider context can be retrieved.
Detailed question: {detailed_question}
Step-back question:
"""

abstarct_question_rewriter_prompt_template = ChatPromptTemplate.from_template(abstarct_question_rewriter_prompt_template)
abstarct_question_rewriter_chain = abstarct_question_rewriter_prompt_template | llm | StrOutputParser()

###############################step 3###################################################### 
# Combining Everything into a Single RAG Chain
#A The context is returned by the retriver after feeding to it the rewritten query
#B This is the original user question

retriever = uk_granular_collection.as_retriever()

rag_prompt_template = """
Given a detailed question and detailed context and abstarct context, answer the question.
If you do not know the answer, just say I do not know.

detailed_context: {detailed_context}
abstract_context: {abstract_context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 
step_back_question_rag_chain = (
    RunnableMap
    (
        {
            "detailed_context": RunnablePassthrough() | detailed_question_rewriter_chain | retriever,
            "abstract_context": RunnablePassthrough() | detailed_question_rewriter_chain | abstarct_question_rewriter_chain | retriever,  # A
            "question": RunnablePassthrough(),  # B
        }
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

user_question = "Tell me some fun things I can do in Cornwall?"

answer = step_back_question_rag_chain.invoke(user_question)
print(answer)