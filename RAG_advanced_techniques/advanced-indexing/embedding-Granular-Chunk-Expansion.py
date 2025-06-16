from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from dotenv import load_dotenv
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document

load_dotenv()  # Load environment variables from .env file

#A Splitter to generate granular chunks from original documents (parsed from web pages)
#B Vector store collection to host child granular chunks
#C Make sure the collection is empty
#D Document store to host expanded chunks
#E Retriever to link parent coarse chunks to child granular chunks
granular_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=500) #A

granular_chunks_collection = Chroma( #B
    collection_name="uk_granular_chunks",
    embedding_function=OpenAIEmbeddings(),
)

granular_chunks_collection.reset_collection() #C

expanded_chunk_store = InMemoryByteStore() #D
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( #E
    vectorstore=granular_chunks_collection,
    byte_store=expanded_chunk_store
)

#A Loader for one destination
#B Documents of one destination 
#C transform HTML docs into clean text docs
#D Split the destination content into granular chunks
#E Iterate over the granular chunks
#F determine the index of the current chunk and its previous and next chunks
#G Assemble the text of the expanded chunk by including the previous and next chunk
#H Generate the ID of the expanded chunk
#I Create the expanded chunk document
#J Link each granular chunk to its related expanded chunk
#K Ingest the granular chunks into the vector store
#L Ingest the expanded chunks into the document store

uk_destination_urls = [
    "https://www.bbc.com/news/articles/cwy04km1zk0o"
]

html2text_transformer = Html2TextTransformer()

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) #A
    html_docs =  html_loader.load() #B
    text_docs = html2text_transformer.transform_documents(html_docs) #C

    granular_chunks = granular_chunk_splitter.split_documents(text_docs) #D

    expanded_chunk_store_items = []
    for i, granular_chunk in enumerate(granular_chunks): #E

        this_chunk_num = i #F
        previous_chunk_num = i-1 #F
        next_chunk_num = i+1 #F
        
        if i==0: #F
            previous_chunk_num = None
        elif i==(len(granular_chunks)-1): #F
            next_chunk_num = None

        expanded_chunk_text = "" #G
        if previous_chunk_num: #G
            expanded_chunk_text += granular_chunks[previous_chunk_num].page_content
            expanded_chunk_text += "\n"

        expanded_chunk_text += granular_chunks[this_chunk_num].page_content #G
        expanded_chunk_text += "\n"

        if next_chunk_num: #G
            expanded_chunk_text += granular_chunks[next_chunk_num].page_content
            expanded_chunk_text += "\n"

        expanded_chunk_id = str(uuid.uuid4()) #H
        expanded_chunk_doc = Document(page_content=expanded_chunk_text) #I

        expanded_chunk_store_item = (expanded_chunk_id, expanded_chunk_doc)
        expanded_chunk_store_items.append(expanded_chunk_store_item)

        granular_chunk.metadata[doc_key] = expanded_chunk_id #J
            
print(f'Ingesting {destination_url}')
multi_vector_retriever.vectorstore.add_documents(granular_chunks) #K
multi_vector_retriever.docstore.mset(expanded_chunk_store_items) #L

retrieved_docs = multi_vector_retriever.invoke("what operation is Israel conducting in Gaza?")
print(retrieved_docs[0].page_content) # Print the content of the first retrieved document