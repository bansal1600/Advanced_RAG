"""
The advantage of this approach is that when a broad query is made, 
the parent chunk is retrieved, providing rich context. 
For more detailed queries, the child embeddings ensure precise 
matching which still lead to the retrieval of the context-rich 
parent chunk. This structure helps the LLM generate more accurate 
and contextually relevant answers.

***We are using the `ParentDocumentRetriever` class from LangChain,
which is designed to retrieve parent documents based on child embeddings***

"""
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer


from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Step 1: Load documents
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

doc_store = InMemoryStore() #E

parent_doc_retriever = ParentDocumentRetriever( #F
    vectorstore=child_chunks_collection,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

#step2: Load documents from a URL
#A Loader for destination web page
#B HTML documents of one destination 
#C Transform HTML docs into clean text deocs
#D Ingest coarse chunks into document store and granular chunks into vector store
uk_destination_urls = [
    "https://www.bbc.com/news/articles/cwy04km1zk0o"
]
html2text_transformer = Html2TextTransformer()  # Initialize the transformer

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) #A
    html_docs =  html_loader.load() #B
    text_docs = html2text_transformer.transform_documents(html_docs) #C

    print(f'Ingesting {destination_url}')
    parent_doc_retriever.add_documents(text_docs, ids=None) #D

print(doc_store.yield_keys())

retrieved_docs_parents = parent_doc_retriever.invoke("forceful operation")
print(retrieved_docs_parents[0].page_content)

print("*"*20)
retrieved_docs_child = child_chunks_collection.similarity_search("forceful operation")
print(retrieved_docs_child[0].page_content)
