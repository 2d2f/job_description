import pandas as pd
import streamlit as st
import io

import os
from src.backend.prompts import MY_PROMPT, GUIDELINES, EXAMPLES

import json
import asyncio

from langchain.chat_models import ChatOpenAI

from src.backend.prompts import MY_PROMPT


# def handle_upload_files():
#     data_source_container = st.container()
#     # container to display infos stored to session state
#     # as it needs to be accessed from submodules
#     uploaded_files = data_source_container.file_uploader("Upload your text file", type=['csv', 'xlsx'], label_visibility="collapsed")
    
#     if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
#         st.session_state["uploaded_files"] = uploaded_files
#         st.session_state["data_source"] = uploaded_files
#         #update_chain()
#     return uploaded_files


# def read_docx(file):
#     """Read a docx file and return the text content."""
#     doc = Document(file)
#     return "\n".join([para.text for para in doc.paragraphs])

def handle_upload_files():

    uploaded_file = st.file_uploader("Choose a file", type=['txt'], label_visibility="collapsed")
    if uploaded_file is not None:
        # Check the file format
        if uploaded_file.name.endswith('.txt'):
            # To read file as string:
            text = str(uploaded_file.read(), "utf-8")
        # elif uploaded_file.name.endswith('.docx'):
        #     text = read_docx(uploaded_file)
    
    if uploaded_file and uploaded_file != st.session_state["uploaded_files"]:
        st.session_state["uploaded_files"] = uploaded_file
        st.session_state["data_source"] = uploaded_file
        #update_chain()
    return uploaded_file

def description_production(intake_doc):
    model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    chain = MY_PROMPT | model
    response = chain.invoke({"intake_doc" : intake_doc, "guidelines" : GUIDELINES, "Examples" : EXAMPLES})

    return response.content


def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        pinecone_key = None
        pinecone_env = None
        if not hasattr(st.session_state, "api_key"):
            st.session_state.api_key = None
        #you can define your API key in .env directly
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            #st.sidebar.success("API key loaded from .env")

        elif st.secrets["OPENAI_API_KEY"] is not None:
            user_api_key = st.secrets["OPENAI_API_KEY"]
            pinecone_key = st.secrets["PINECONE_API_KEY"]
            pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]
            #st.sidebar.success("API key loaded")

        else:
            if st.session_state.api_key is not None:
                user_api_key = st.session_state.api_key
                st.sidebar.success("API key loaded from previous input")
            else:
                user_api_key = st.sidebar.text_input(
                    label="#### Your OpenAI API key", placeholder="sk-...", type="password"
                )
                if user_api_key:
                    st.session_state.api_key = user_api_key

        return user_api_key, pinecone_key, pinecone_env

