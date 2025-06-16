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
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

#step1: create parent coarse chunks and child granular chunks
#A Splitter to generate parent coarse chunks from original documents (parsed from web pages)
#B Vector store collection to host child granular chunks
#C Make sure the collection is empty
#D Document store to host parent coarse chunks
#E Retriever to link parent coarse chunks to child granular chunks

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000) #A

summaries_collection = Chroma( #B
    collection_name="uk_summaries",
    embedding_function=OpenAIEmbeddings(),
)

summaries_collection.reset_collection() #C

doc_byte_store = InMemoryByteStore() #D
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( #E
    vectorstore=summaries_collection,
    byte_store=doc_byte_store
)

# step2 : Setting Up the Summarization Chain
#A Grab the text content from the document
#B Instantiate a prompt asking to generate summary of the provided text
#C Send the LLM the instantiated prompt 
#D Extract the summary text from the response

llm = ChatOpenAI(model="gpt-4o-mini")

summarization_chain = (
    {"document": lambda x: x.page_content} #A
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{document}") #B
    | llm
    | StrOutputParser())

# step3 : Load documents from a URL
#A Loader for one destination
#B Documents of one destination 
#C transform HTML docs into clean text docs
#D Split the destination content into coarse chunks
#E Iterate over the coarse chunks
#F Generate a summary for the coarse chunk thorugh the summarization chain
#G Link each summary to its related coarse chunk
#H Ingest the summaries into the vector store
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

coarse_chunks_ids = []
for _ in coarse_chunks:
    coarse_chunks_ids.append(str(uuid.uuid4))

all_summaries = []
for i, coarse_chunk in enumerate(coarse_chunks): #E
        coarse_chunk_id = coarse_chunks_ids[i]
        summary_text =  summarization_chain.invoke(coarse_chunk) #F
        summary_doc = Document(page_content=summary_text, metadata={doc_key: coarse_chunk_id})
        all_summaries.append(summary_doc) #G

print(f'Ingesting {destination_url}')
multi_vector_retriever.vectorstore.add_documents(all_summaries) #H
multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks))) #I

retrieved_docs_parents = multi_vector_retriever.invoke("forceful operation")
print(retrieved_docs_parents[0].page_content)

print("*"*20)
retrieved_docs_child = summaries_collection.similarity_search("forceful operation")
print(retrieved_docs_child[0].page_content)
