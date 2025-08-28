import streamlit as st
from utils import *
import constants
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure a default USER_AGENT is set to identify requests politely
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "ragbot-app/1.0 (contact: admin@example.com)"

# Creating Session State Variable
if 'HuggingFace_API_Key' not in st.session_state:
    st.session_state['HuggingFace_API_Key'] =''
if 'Pinecone_API_Key' not in st.session_state:
    st.session_state['Pinecone_API_Key'] =''
if 'Google_API_Key' not in st.session_state:
    st.session_state['Google_API_Key'] =''

#
st.title('🤖 AI RAG Assistant For Website (Gemini)') 

#********SIDE BAR Funtionality started********

# Sidebar to capture the API keys
st.sidebar.title("😎🗝️")
st.session_state['HuggingFace_API_Key']= st.sidebar.text_input("What's your HuggingFace API key?",type="password")
st.session_state['Pinecone_API_Key']= st.sidebar.text_input("What's your Pinecone API key?",type="password")
st.session_state['Google_API_Key']= st.sidebar.text_input("What's your Google API key?",type="password")

#Recent changes by langchain team, expects ""PINECONE_API_KEY" environment variable for Pinecone usage! So we are creating it here
os.environ["PINECONE_API_KEY"] = st.session_state['Pinecone_API_Key']
os.environ["GOOGLE_API_KEY"] = st.session_state['Google_API_Key']

load_button = st.sidebar.button("Load data to Pinecone", key="load_button")

#If the bove button is clicked, pushing the data to Pinecone...
if load_button:
    #Proceed only if API keys are provided
    if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['Pinecone_API_Key']!="" :

        #Fetch data from multiple sources (website + local files)
        site_data = get_documents_from_sources(constants.DOCUMENT_SOURCES)
        st.write(f"Data pull done... Loaded {len(site_data)} documents")

        #Split data into chunks
        chunks_data=split_data(site_data)
        st.write("Spliting data done...")

        #Creating embeddings instance
        embeddings=create_embeddings()
        st.write("Embeddings instance creation done...")

        #Push data to Pinecone
        
        push_to_pinecone(st.session_state['Pinecone_API_Key'],constants.PINECONE_ENVIRONMENT,constants.PINECONE_INDEX,embeddings,chunks_data)
        st.write("Pushing data to Pinecone done...")

        st.sidebar.success("Data pushed to Pinecone successfully!")
    else:
        st.sidebar.error("Ooopssss!!! Please provide API keys.....")

#********SIDE BAR Funtionality ended*******

#Captures User Inputs
prompt = st.text_input('How can I help you my friend ❓',key="prompt")  # The box for the text prompt
document_count = st.slider('No.Of links to return 🔗 - (0 LOW || 5 HIGH)', 0, 5, 2,step=1)

submit = st.button("Ask AI Assistant") 


if submit:
    #Proceed only if API keys are provided
    if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['Pinecone_API_Key']!="" and st.session_state['Google_API_Key']!="" :

        #Creating embeddings instance
        embeddings=create_embeddings()
        st.write("Embeddings instance creation done...")

        #Pull index data from Pinecone
        index=pull_from_pinecone(st.session_state['Pinecone_API_Key'],constants.PINECONE_ENVIRONMENT,constants.PINECONE_INDEX,embeddings)
        st.write("Pinecone index retrieval done...")

        #Fetch relavant documents from Pinecone index
        relavant_docs=get_similar_docs(index,prompt,document_count)
        st.write("Relevant documents retrieved...")

        # Generate RAG response
        llm = create_llm()
        if llm:
            st.write("Generating AI response...")
            rag_response = generate_rag_response(llm, relavant_docs, prompt)
            
            # Display AI-generated answer
            st.success("🤖 AI Assistant Response:")
            st.write(rag_response["answer"])
            
            # Display sources
            if rag_response["sources"]:
                st.info("📚 Sources:")
                for i, source in enumerate(rag_response["sources"], 1):
                    st.write(f"{i}. {source}")
            
            # Display raw context (collapsible)
            with st.expander("🔍 View Retrieved Context"):
                for i, doc in enumerate(rag_response["context_docs"], 1):
                    st.write(f"**Document {i}:**")
                    st.write(f"**Content:** {doc.page_content}")
                    st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                    st.write("---")
        else:
            st.error("Failed to create Gemini LLM instance. Please check your Google API key.")
    else:
        st.sidebar.error("Ooopssss!!! Please provide all required API keys.....")


   
