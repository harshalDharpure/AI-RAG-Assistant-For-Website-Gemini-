import streamlit as st
from utils import *
import constants

# Creating Session State Variable
if 'HuggingFace_API_Key' not in st.session_state:
    st.session_state['HuggingFace_API_Key'] =''
if 'Pinecone_API_Key' not in st.session_state:
    st.session_state['Pinecone_API_Key'] =''


#
st.title('🤖 AI Assistance For Website') 

#********SIDE BAR Funtionality started********

# Sidebar to capture the API keys
st.sidebar.title("😎🗝️")
st.session_state['HuggingFace_API_Key']= st.sidebar.text_input("What's your HuggingFace API key?",type="password")
st.session_state['Pinecone_API_Key']= st.sidebar.text_input("What's your Pinecone API key?",type="password")

#Recent changes by langchain team, expects ""PINECONE_API_KEY" environment variable for Pinecone usage! So we are creating it here
import os
os.environ["PINECONE_API_KEY"] = st.session_state['Pinecone_API_Key']


source = st.sidebar.radio("Data source", ["Sitemap URL","Local folder"], index=0)
data_dir = st.sidebar.text_input("Local folder path (for PDFs/TXTs)", value="data")
load_button = st.sidebar.button("Load data to Pinecone", key="load_button")

#If the bove button is clicked, pushing the data to Pinecone...
if load_button:
    #Proceed only if API keys are provided
    if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['Pinecone_API_Key']!="" :

        #Fetch data based on selected source
        if source=="Sitemap URL":
            site_data=get_website_data(constants.WEBSITE_URL)
        else:
            site_data=get_local_data(data_dir)
        st.write("Data pull done...")

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

submit = st.button("Search") 


if submit:
    #Proceed only if API keys are provided
    if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['Pinecone_API_Key']!="" :

        #Creating embeddings instance
        embeddings=create_embeddings()
        st.write("Embeddings instance creation done...")

        #Pull index data from Pinecone
        index=pull_from_pinecone(st.session_state['Pinecone_API_Key'],constants.PINECONE_ENVIRONMENT,constants.PINECONE_INDEX,embeddings)
        st.write("Pinecone index retrieval done...")

        #Fetch relavant documents from Pinecone index
        relavant_docs=get_similar_docs(index,prompt,document_count)

        if not relavant_docs:
            st.warning("No relevant documents found. Try a different question or load data first.")
        else:
            # Build prompt and generate simple snippet-based answer (no external LLM)
            rag_prompt, sources = build_rag_prompt(prompt, relavant_docs)
            answer = generate_simple_answer(prompt, relavant_docs)

            # Show final answer with citations
            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Sources")
                for i, src in enumerate(sources, start=1):
                    st.write(f"{i}. {src}")
       


    else:
        st.sidebar.error("Ooopssss!!! Please provide API keys.....")


   
