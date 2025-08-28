# WEBSITE_URL="https://www.decode.com/wp-sitemap-posts-post-1.xml"
# WEBSITE_URL="https://geekflare.com/wp-sitemap-posts-post-1.xml"
WEBSITE_URL="https://www.indiancounsellingservices.com/job-listing/sitemap-1.xml"
PINECONE_ENVIRONMENT="us-east-1"
PINECONE_INDEX="ragbot"

# Document Sources Configuration
DOCUMENT_SOURCES = {
    "website_url": WEBSITE_URL,
    "website_limit": 5,  # Number of website pages to load
    "local_path": "./documents",  # Path to local documents folder
    "file_types": [".txt", ".pdf", ".docx", ".md"]  # Supported file types
}

# RAG Configuration
LLM_MODEL = "gemini-pro"  # Google Gemini model
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions based on the provided context from a website.

Context Information:
{context}

User Question: {question}

Please provide a comprehensive and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so. Always cite the source links when possible.

Answer:"""