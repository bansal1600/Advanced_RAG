##########ONE TIME RUN!!!! to create a Chroma DB from web pages##########

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


PERSIST_DIR = "./chroma_db"

# Check if DB already exists
if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
    print("Loading existing Chroma DB...")
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR,
    )
else:
    print("Creating new Chroma DB...")
    urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

    docs_lists = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs_lists.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_lists)

    print(f"Number of documents loaded: {len(doc_splits)}")

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        collection_name="rag-chroma",
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()