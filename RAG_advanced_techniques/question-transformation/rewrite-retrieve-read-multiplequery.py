from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate

from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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


#step 2: Setting up Multi Query retriever
multi_query_gen_prompt_template = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}
"""

multi_query_gen_prompt = ChatPromptTemplate.from_template(multi_query_gen_prompt_template)

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  


questions_parser = LineListOutputParser()

llm = ChatOpenAI(model="gpt-4o-mini")
multi_query_gen_chain = multi_query_gen_prompt | llm | questions_parser

user_question = " Tell me some fun things I can do in Cornwall."
multiple_queries = multi_query_gen_chain.invoke(user_question)

basic_retriever = uk_granular_collection.as_retriever()
multi_query_retriever = MultiQueryRetriever(
    retriever=basic_retriever, llm_chain=multi_query_gen_chain, 
    parser_key="lines" #A
)  

user_question = "Tell me some fun things I can do in Cornwall"
retrieved_docs = multi_query_retriever.invoke(user_question)
print(retrieved_docs)