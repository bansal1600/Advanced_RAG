from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_community.document_transformers import Html2TextTransformer
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# step1: create parent coarse chunks and child granular chunks
#A Splitter to generate parent coarse chunks from original documents (parsed from web pages)
#B Vector store collection to host child granular chunks
#C Make sure the collection is empty
#D Document store to host parent coarse chunks
#E Retriever to link parent coarse chunks to child granular chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000) #A

hypotetical_questions_collection = Chroma( #B
    collection_name="uk_hypotetical_questions",
    embedding_function=OpenAIEmbeddings(),
)

hypotetical_questions_collection.reset_collection() #C

doc_byte_store = InMemoryByteStore() #D
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( #E
    vectorstore=hypotetical_questions_collection,
    byte_store=doc_byte_store
)

#step2 : Setting Up the Hypothetical Question Generation Chain
class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions for given text."""
    questions: List[str] = Field(..., description="List of hypotetical questions for given text")

llm_with_structured_output = (
    ChatOpenAI(model="gpt-4o-mini")
    .with_structured_output(HypotheticalQuestions)
)

#A Grab the text content from the document
#B Instantiate a prompt asking to generate 4 hypotetical questions on the provided text
#C Invoke the LLM configured to return an object containing the questions as a typed list of strings
#D Grab the list of questions from the response
hypotetical_questions_chain = (
    {"document_text": lambda x: x.page_content} #A
    | ChatPromptTemplate.from_template( #B
        "Generate a list of exactly 4 hypothetical questions that the below text could be used to answer:\n\n{document_text}"
    )
    | llm_with_structured_output #C
    | (lambda x: x.questions) #D
)

# step3 : Ingesting Coarse Chunks and Related Hypothetical Questions
#A Loader for one destination
#B Documents of one destination 
#C transform HTML docs into clean text docs
#D Split the destination content into coarse chunks
#E Iterate over the coarse chunks
#F Generate a list of hypothetical questions for the coarse chunk through the question generation chain
#G Link each hypothetical question to its related coarse chunk
#H Ingest the hypothetical questions into the vector store
#I Ingest the coarse chunks into the document store
uk_destination_urls = [
    "https://www.bbc.com/news/articles/cwy04km1zk0o"
]
html2text_transformer = Html2TextTransformer()  # Initialize the transformer

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) #A
    html_docs =  html_loader.load() #B
    text_docs = html2text_transformer.transform_documents(html_docs) #C

    coarse_chunks = parent_splitter.split_documents(text_docs) #D

    coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
    all_hypotetical_questions = []
    for i, coarse_chunk in enumerate(coarse_chunks): #E
        
        coarse_chunk_id = coarse_chunks_ids[i]
            
        hypotetical_questions = hypotetical_questions_chain.invoke(coarse_chunk) #F
        hypotetical_questions_docs = [
            Document(page_content=question, metadata={doc_key: coarse_chunk_id})
                                              for question in hypotetical_questions
                                              ] #G

        all_hypotetical_questions.extend(hypotetical_questions_docs)

print(f'Ingesting {destination_url}')
multi_vector_retriever.vectorstore.add_documents(all_hypotetical_questions) #H
multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks))) #I

hypothetical_question_docs_only = hypotetical_questions_collection.similarity_search("what operation is Israel conducting in Gaza?")
print(hypothetical_question_docs_only) 

retrieved_docs = multi_vector_retriever.invoke("what operation is Israel conducting in Gaza?")
print(retrieved_docs[0].page_content) # Print the content of the first retrieved document