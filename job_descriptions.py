# If not on macOS, uncomment the three line herunder (pysqlite3 should be installed)
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import importlib
import sys
import pandas as pd
import streamlit as st
from src.backend.layout import Layout

from src.backend.sidebar import Sidebar
from src.streamlit.helper import initialize_session_state

from  src.backend.utils_descriptions import description_production, load_api_key, handle_upload_files

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


st.set_page_config(layout="wide", page_title="Loop Earplugs | Job Description")

layout, sidebar = Layout(), Sidebar()

st.markdown(
            f"""
            <h1 style='text-align: center;'> Job Description </h1>
            """,
            unsafe_allow_html=True,
        )


def init_agent():
    llm = OpenAI(temperature=0)
    review_content = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=st.session_state["vector_db"].as_retriever()
    )
    tools = [
        Tool(
            name="Review Content QA System",
            func=review_content.run,
            description="useful for when you need to answer questions about the content of the product reviews. Input should be a fully formed question.",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return agent

@st.cache_data
def get_answer(_agent, query):
    return _agent.run(query)

# Define all containers upfront to ensure app UI consistency
# container for all data source widgets:
# load existing vector stores, upload files, or enter any datasource string
#data_source_container = st.container()
# container to display infos stored to session state
# as it needs to be accessed from submodules
#st.session_state["info_container"] = st.empty()

user_api_key, PINECONE_API_KEY, PINECONE_ENV = load_api_key()
os.environ["OPENAI_API_KEY"] = user_api_key

initialize_session_state()

st.session_state["uploaded_file"] = None

if not user_api_key:
    layout.show_api_key_missing()

else:
    st.session_state.setdefault("reset_chat", False)
    
    st.write("**1. Drag and drop one Excel sheet of Product reviews.**")
    
    st.session_state["uploaded_files"] = handle_upload_files()

    if st.session_state["uploaded_files"]:
        response = []
        for i in range(3) : 
            st.write("Call to chatgpt")
            resp = description_production(st.session_state["uploaded_files"])
            response.append(resp)
            st.write(response[i])
        
        st.download_button("Download all the descriptions")...
