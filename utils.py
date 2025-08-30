from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import asyncio
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Tuple
import os




#Function to fetch data from website
# https://python.langchain.com/docs/integrations/document_loaders/sitemap/
# here we are passing a limit value of 5 so that we dont end up getting every link as its just a demo project
def get_website_data(sitemap_url, limit=5):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(
    sitemap_url
    )

    docs = loader.load()

    return docs[:limit]

#Function to fetch local documents from a directory (supports PDF and TXT)
def get_local_data(data_dir):

    # Load PDFs
    pdf_loader = DirectoryLoader(
    data_dir,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()

    # Load TXTs
    txt_loader = DirectoryLoader(
    data_dir,
    glob="**/*.txt",
    loader_cls=TextLoader
    )
    txt_docs = txt_loader.load()

    return pdf_docs + txt_docs

#Function to split data into smaller chunks
def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

#Function to create embeddings instance
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Removed external LLM dependency for a simpler, offline-friendly flow

#Function to build a simple RAG prompt from retrieved documents
def build_rag_prompt(query: str, docs: List) -> Tuple[str, List[str]]:

    sources = []
    context_chunks = []
    for doc in docs:
        context_chunks.append(doc.page_content)
        # Capture source if present
        src = None
        try:
            src = doc.metadata.get("source")
        except Exception:
            src = None
        if src:
            sources.append(src)

    # Deduplicate sources while preserving order
    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    context = "\n\n".join(context_chunks)
    system_instructions = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer cannot be found in the context, say you don't know. Be concise."
    )
    prompt = (
        f"{system_instructions}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    return prompt, unique_sources

#Very simple answer generator without external LLM: extract relevant sentences from retrieved docs
def generate_simple_answer(query: str, docs: List, max_sentences: int = 6) -> str:

    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    scored_sentences = []

    for doc in docs:
        text = doc.page_content
        # naive sentence split
        sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
        for s in sentences:
            s_lower = s.lower()
            score = sum(1 for t in query_terms if t in s_lower)
            if score > 0:
                scored_sentences.append((score, s))

    if not scored_sentences:
        # fallback: take the first few lines of the top document(s)
        snippets = []
        for doc in docs:
            snippet = doc.page_content.strip().split('\n')[:3]
            snippets.extend(snippet)
            if len(snippets) >= max_sentences:
                break
        return '\n'.join(snippets[:max_sentences])

    # sort by score descending, keep unique sentences while preserving order
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    result = []
    for _, s in scored_sentences:
        if s not in seen:
            result.append(s)
            seen.add(s)
        if len(result) >= max_sentences:
            break

    return '. '.join(result) + '.'

#Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    Pinecone(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    #PineconeStore is an alias name of Pinecone class, please look at the imports section at the top :)
    index =  PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    return index

#Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    Pinecone(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    #PineconeStore is an alias name of Pinecone class, please look at the imports section at the top :)
    index = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return index

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


