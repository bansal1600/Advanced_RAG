from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
import uuid
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

#step1: create parent coarse chunks and child granular chunks
#A Splitter to generate parent coarse chunks from original documents (parsed from web pages)
#B Splitter to generate child granular chunks from parent coarse chunks
#C Vector store collection to host child granular chunks
#D Make sure the collection is empty
#E Document store to host parent coarse chunks
#F Retriever to link parent coarse chunks to child granular chunks

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000) #A
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500) #B

child_chunks_collection = Chroma( #C
    collection_name="uk_child_chunks",
    embedding_function=OpenAIEmbeddings(),
)

child_chunks_collection.reset_collection() #D

doc_byte_store = InMemoryByteStore() #E
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( #F
    vectorstore=child_chunks_collection,
    byte_store=doc_byte_store
)

# step2 : Load documents from a URL

#step2: Load documents from a URL
#A Loader for one destination
#B Documents of one destination 
#C transform HTML docs into clean text docs
#D Split the destination content into parent coarse chunks
#E Iterate over the parent coarse chunks
#F Create child granular chunks form each parent coarse chunk
#G Link each child granular chunk to its parent coarse chunk
#H Ingest the child granular chunks into the vector store
#I Ingest the parent coarse chunks into the document store

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
    
all_granular_chunks = []
for i, coarse_chunk in enumerate(coarse_chunks): #E
    coarse_chunk_id = coarse_chunks_ids[i]
    granular_chunks = child_splitter.split_documents([coarse_chunk]) #F
    for granular_chunk in granular_chunks:
        granular_chunk.metadata[doc_key] = coarse_chunk_id #G
    all_granular_chunks.extend(granular_chunks)

print(f'Ingesting {destination_url}')
multi_vector_retriever.vectorstore.add_documents(all_granular_chunks) #H
multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks))) #I

retrieved_docs_parents = multi_vector_retriever.invoke("forceful operation")
print(retrieved_docs_parents[0].page_content)

print("*"*20)
retrieved_docs_child = child_chunks_collection.similarity_search("forceful operation")
print(retrieved_docs_child[0].page_content)