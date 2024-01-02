import os
import asyncio
import importlib
import sys
import pandas as pd
import streamlit as st
from src.backend.layout import Layout

from src.backend.sidebar import Sidebar
from src.streamlit.helper import initialize_session_state
from src.backend.utils_product_review import concatenate_reviews, process_uploaded_file, first_insights, load_api_key, handle_upload_files
from src.backend.utils_topic_modelling import get_representative_dataset, get_topics_list, classify_reviews, cached_classify_reviews

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import seaborn as sns
import matplotlib.pyplot as plt


def display_markdown():
    st.markdown(
        """
        <h1 style='text-align: center;'> Product Reviews Analyzer with detailed insights </h1>
        """,
        unsafe_allow_html=True,
    )


def handle_api_key():
    user_api_key, PINECONE_API_KEY, PINECONE_ENV = load_api_key()
    os.environ["OPENAI_API_KEY"] = user_api_key
    if not user_api_key:
        layout.show_api_key_missing()

def handle_upload_files():
    st.write("**1. Drag and drop one Excel sheet of Product reviews.**")
    uploaded_files = st.file_uploader("", type=["xlsx"])
    return uploaded_files

def process_and_display_file(uploaded_files):
    # File processing and display logic
    ...

