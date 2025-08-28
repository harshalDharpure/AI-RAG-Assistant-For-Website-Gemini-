from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import asyncio
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from pathlib import Path

# Optional loaders (graceful fallback if not installed)
try:
    from langchain_community.document_loaders import TextLoader
except Exception:
    TextLoader = None

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain_community.document_loaders import Docx2txtLoader
except Exception:
    Docx2txtLoader = None

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
except Exception:
    UnstructuredMarkdownLoader = None



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

#Function to create LLM instance for RAG
def create_llm():
    """Create and return a Gemini LLM instance for generating responses"""
    model_from_env = os.getenv("GEMINI_MODEL")
    preferred_model = model_from_env if model_from_env else "gemini-1.5-flash"
    fallback_models = ["gemini-1.5-pro", "gemini-1.0-pro"]

    # Build ordered list to try
    model_candidates = [preferred_model] + [m for m in fallback_models if m != preferred_model]

    last_error = None
    for model_name in model_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_output_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            # Touch a trivial call to validate credentials/model lazily if needed could be expensive; skip and rely on first call
            return llm
        except Exception as e:
            last_error = e
            print(f"Failed initializing Gemini model '{model_name}': {e}")
            continue

    print(f"Error creating Gemini LLM after trying {model_candidates}: {last_error}")
    return None

#Function to generate RAG response
def generate_rag_response(llm, context_docs, user_question):
    """Generate a response using RAG approach"""
    try:
        # Prepare context from retrieved documents
        context_text = ""
        source_links = []
        
        for doc in context_docs:
            context_text += f"\n\nContent: {doc.page_content}\n"
            if 'source' in doc.metadata:
                source_links.append(doc.metadata['source'])
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=os.getenv("RAG_PROMPT_TEMPLATE", 
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        )
        
        # Create LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Generate response
        response = llm_chain.run({
            "context": context_text,
            "question": user_question
        })
        
        return {
            "answer": response,
            "sources": source_links,
            "context_docs": context_docs
        }
        
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return {
            "answer": f"Sorry, I encountered an error while generating the response: {str(e)}",
            "sources": [],
            "context_docs": []
        }

#Function to load local documents
def load_local_documents(documents_path, file_types=None):
    """Load documents from local directory"""
    # Filter file types to those we can actually load
    supported_types = []
    if TextLoader is not None:
        supported_types.append('.txt')
    if PyPDFLoader is not None:
        supported_types.append('.pdf')
    if Docx2txtLoader is not None:
        supported_types.append('.docx')
    if UnstructuredMarkdownLoader is not None:
        supported_types.append('.md')

    if file_types is None:
        file_types = supported_types
    else:
        file_types = [ext for ext in file_types if ext in supported_types]

    documents = []
    documents_path = Path(documents_path)
    
    if not documents_path.exists():
        print(f"Path {documents_path} does not exist")
        return documents
    
    for file_path in documents_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in file_types:
            try:
                loader = None
                if file_path.suffix.lower() == '.txt' and TextLoader is not None:
                    loader = TextLoader(str(file_path))
                elif file_path.suffix.lower() == '.pdf' and PyPDFLoader is not None:
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() == '.docx' and Docx2txtLoader is not None:
                    loader = Docx2txtLoader(str(file_path))
                elif file_path.suffix.lower() == '.md' and UnstructuredMarkdownLoader is not None:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                
                if loader is None:
                    print(f"Skipping unsupported type or missing loader for: {file_path.suffix} ({file_path})")
                    continue
                
                docs = loader.load()
                # Add source metadata
                for doc in docs:
                    doc.metadata['source'] = str(file_path)
                documents.extend(docs)
                print(f"Loaded: {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents

#Function to get documents from multiple sources
def get_documents_from_sources(sources):
    """Get documents from multiple sources (website + local files)"""
    all_docs = []
    
    # Load website documents if URL is provided
    if sources.get('website_url'):
        try:
            website_docs = get_website_data(sources['website_url'], sources.get('website_limit', 5))
            all_docs.extend(website_docs)
            print(f"Loaded {len(website_docs)} documents from website")
        except Exception as e:
            print(f"Error loading website documents: {e}")
    
    # Load local documents if path is provided
    if sources.get('local_path'):
        try:
            local_docs = load_local_documents(
                sources['local_path'], 
                sources.get('file_types', ['.txt', '.pdf', '.docx', '.md'])
            )
            all_docs.extend(local_docs)
            print(f"Loaded {len(local_docs)} documents from local path")
        except Exception as e:
            print(f"Error loading local documents: {e}")
    
    return all_docs


