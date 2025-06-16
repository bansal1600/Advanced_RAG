#refer this article https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-f9c82290b0d1

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncHtmlLoader

load_dotenv()  # Load environment variables from .env file

#Loading HTML Content with AsyncHtmlLoader
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
html_section_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

def split_docs_into_granular_chunks(docs):
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content #A  
        temp_chunks = html_section_splitter.split_text(html_string) #B  
        all_chunks.extend(temp_chunks)
    return all_chunks

html2text_transformer = Html2TextTransformer()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=300
)
def split_docs_into_coarse_chunks(docs):
    text_docs = html2text_transformer.transform_documents(docs) #A  
    coarse_chunks = text_splitter.split_documents(text_docs) #B
    return coarse_chunks

uk_granular_collection = Chroma( #A
    collection_name="uk_granular",
    embedding_function=OpenAIEmbeddings(),
)
uk_granular_collection.reset_collection() #B

uk_coarse_collection = Chroma( #A
    collection_name="uk_coarse",
    embedding_function=OpenAIEmbeddings(),
)
uk_coarse_collection.reset_collection() #B

uk_destinations = [
    "Cornwall"
    # , "North_Cornwall", "South_Cornwall", "West_Cornwall", 
    # "Tintagel", "Bodmin", "Wadebridge", "Penzance", "Newquay",
    # "St_Ives", "Port_Isaac", "Looe", "Polperro", "Porthleven",
    # "East_Sussex", "Brighton", "Battle", "Hastings_(England)", 
    # "Rye_(England)", "Seaford", "Ashdown_Forest"
]

wikivoyage_root_url = r"https://en.wikivoyage.org/wiki"

uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) #C
    docs = html_loader.load() #D
    
    granular_chunks = split_docs_into_granular_chunks(docs)
    uk_granular_collection.add_documents(documents=granular_chunks)

    coarse_chunks = split_docs_into_coarse_chunks(docs)
    uk_coarse_collection.add_documents(documents=coarse_chunks)

#A Create a Chorma DB collection
#B Reset the collection in case it already exists 
#C Loader for one destination
#D Documents of one destination
# You can now perform both granular and coarse searches:
granular_results = uk_granular_collection.similarity_search(query="Events or festivals in East Sussex", k=4)
for doc in granular_results:
    print(doc)

coarse_results = uk_coarse_collection.similarity_search(query="Events or festivals in East Sussex", k=4)
for doc in coarse_results:
    print(doc)