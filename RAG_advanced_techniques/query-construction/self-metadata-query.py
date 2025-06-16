"""
Self-Querying (Metadata Query Enrichment):
A vector store typically indexes document chunks by embedding for dense search, but it can also use keyword-based indexing in a few ways:
1. Explicit Metadata Tags: You can add metadata to each chunk, such as the timestamp, filename or URL, topic, and keywords. These keywords can come from user input or ones you assign manually.
2. Keyword Extraction via Algorithm: Use algorithms like TF-IDF (Term Frequency-Inverse Document Frequency) or its extension, BM25, to identify relevant keywords for each chunk based on word frequency and importance.
3. Keyword Suggestions from the LLM: You can ask the LLM to generate keywords for tagging each chunk.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

uk_with_metadata_collection = Chroma(
    collection_name="uk_with_metadata_collection",
    embedding_function=OpenAIEmbeddings())

#A in case it already exists
uk_with_metadata_collection.reset_collection() #A


# Define Ingestion Content and Splitting Strategy
#A Instantiate a relatively fine-chunk splitting strategy
#B Transform HTML docs into clean text docs
#C Prepare metadata to be imported: Url, UK Destination and UK Region

html2text_transformer = Html2TextTransformer()

text_splitter = RecursiveCharacterTextSplitter( #A
    chunk_size=1000, chunk_overlap=100
)

def split_docs_into_chunks(docs):
    text_docs = html2text_transformer.transform_documents(docs) #B
    chunks = text_splitter.split_documents(text_docs)

    return chunks

uk_destinations = [
    ("Cornwall", "Cornwall")
]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"

uk_destination_url_with_metadata = [ #C 
    ( f'{wikivoyage_root_url}/{destination}', destination, region)
    for destination, region in uk_destinations]

print(uk_destination_url_with_metadata)

# Ingest Content with Metadata
#A Loader for one destination
#B Documents (chunks) related to one destination

for (url, destination, region) in uk_destination_url_with_metadata:
    html_loader = AsyncHtmlLoader(url) #A
    docs =  html_loader.load() #B
    
    docs_with_metadata = []
    for d in docs:
        docs_with_metadata.append(Document(page_content=d.page_content,
        metadata = {
            'source': url,
            'destination': destination,
            'region': region}))
             
    chunks = split_docs_into_chunks(docs_with_metadata)

    print(f'Importing: {destination}')
    uk_with_metadata_collection.add_documents(documents=chunks)

"""
Q & A on a Metadata-Enriched Collection
There are three ways to query metadata-enriched content:

1. Explicit Metadata Filters: Specify the metadata filter manually.

2. SelfQueryRetriever: Automatically generate the metadata filter using the SelfQueryRetriever.

3. Structured LLM Function Call: Infer the metadata filter with a structured call to an LLM function.
"""

######### Querying with an Explicit Metadata Filter############

question =  "Events or festivals"
metadata_retriever = uk_with_metadata_collection.as_retriever(search_kwargs={'k':2, 'filter':{'destination': 'Cornwall'}})

result_docs = metadata_retriever.invoke(question)

print(result_docs[0])
print('*' * 100)

############ Automatically Generating Metadata Filters with SelfQueryRetriever ##############################

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever #A
from langchain_openai import ChatOpenAI
#A this requires pip install lark

# Next, define the metadata attributes to infer from the question:
metadata_field_info = [
    AttributeInfo(
        name="destination",
        description="The specific UK destination to be searched",
        type="string",
    ),
    AttributeInfo(
        name="region",
        description="The name of the UK region to be searched",
        type="string",
    )
]

question = "Tell me about events or festivals in the UK town of Cornwall"

document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(model="gpt-4o-mini")

self_query_retriever = SelfQueryRetriever.from_llm(
    llm, uk_with_metadata_collection, question, metadata_field_info, verbose=True
)

result_docs = self_query_retriever.invoke(question)

print(result_docs[0])
print('*' * 100)


##############Generating Metadata Filters with an LLM Function Call##########
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain_community.query_constructors.chroma import ChromaTranslator

class DestinationSearch(BaseModel):
    destination: str = Field(description="The specific town, city, or place being asked about.")
    region: str = Field(description="The larger region or country relevant to the query.")
    content_search: str = Field(description="The actual topic or subject of interest like events, festivals, attractions, etc.")


def build_filter(destination_search: DestinationSearch):
    comparisons = []

    destination = destination_search.destination #A
    region = destination_search.region #A
    
    if destination and destination != '': #B
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="destination",
                value=destination,
            )
        )
    if region and region != '': #C
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="region",
                value=region,
            )
        )    

    search_filter = Operation(operator=Operator.AND, arguments=comparisons) #D

    chroma_filter = ChromaTranslator().visit_operation(search_filter) #E
        
    return chroma_filter
#A Get destination and region from the structured query
#B If the destination exists, create an 'equality' operation
#C If the region exists, create an 'equality' operation
#D Create a combined search filter
#E Transform the filter into Chroma format


system_message = """You are an expert at converting user questions into vector database queries. \
You have access to a database of tourist destinations. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(DestinationSearch)
query_generator = prompt | structured_llm

question = "Tell me about events or festivals in Cornwall"
structured_query =query_generator.invoke(question)

search_filter = build_filter(structured_query)
search_query = structured_query.content_search

metadata_retriever = uk_with_metadata_collection.as_retriever(search_kwargs={'k':3, 'filter': search_filter})

answer = metadata_retriever.invoke(search_query)

print(answer)
print('*' * 100)