import os
import pandas as pd
import streamlit as st
import pdfplumber
import pandas as pd
#import snowflake.connector
from datetime import datetime

import concurrent.futures
from io import BytesIO
from src.backend.chatbot import Chatbot
from src.backend.embedder import Embedder
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from src.streamlit.helper import (
    update_chain,
)
from src.backend.logging import logger

# TODO :  need to improve this class (maybe no need of main class)
# Maybe a class for resume analysis is necessary
# Big functions must be lower into smaller functions calling each other
class Utilities:

    @staticmethod
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

    
    @staticmethod
    def handle_upload(file_types):
        """
        Handles and display uploaded_file
        :param file_types: List of accepted file types, e.g., ["csv", "pdf", "txt"]
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=file_types, label_visibility="collapsed")
        if uploaded_file is not None:

            def show_csv_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                uploaded_file.seek(0)
                shows = pd.read_csv(uploaded_file)
                file_container.write(shows)

            def show_pdf_file(uploaded_file):
                file_container = st.expander("Your PDF file :")
                with pdfplumber.open(uploaded_file) as pdf:
                    pdf_text = ""
                    for page in pdf.pages:
                        pdf_text += page.extract_text() + "\n\n"
                file_container.write(pdf_text)
            
            def show_txt_file(uploaded_file):
                file_container = st.expander("Your TXT file:")
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8")
                file_container.write(content)
            
            def get_file_extension(uploaded_file):
                return os.path.splitext(uploaded_file)[1].lower()
            
            file_extension = get_file_extension(uploaded_file.name)

            # Show the contents of the file based on its extension
            if file_extension == ".csv" :
                show_csv_file(uploaded_file)
            if file_extension== ".pdf" : 
                show_pdf_file(uploaded_file)
            elif file_extension== ".txt" : 
                show_txt_file(uploaded_file)

        else:
            st.session_state["reset_chat"] = True

        #print(uploaded_file)
        return uploaded_file
    
    @staticmethod
    def handle_text_load(label = "job"):
        text_input = st.sidebar.text_input('Enter the url leading to the job description.', placeholder="https://..." )
        def is_website(text):
            return text[:8] == "https://"
        
        def show_txt_file(content):
            st.write("Your job description url: "+content)
        if is_website(text_input):
            show_txt_file(text_input)
            return text_input
        else:
            st.write("Enter a valid url for the job description.")
            return False
        
    @staticmethod
    def handle_upload_files(file_types = "pdf"):
        data_source_container = st.container()
        # container to display infos stored to session state
        # as it needs to be accessed from submodules
        uploaded_files = data_source_container.file_uploader("**2. Drag and drop the candidate resume. Bulk upload of several resumes from Greenhouse is possible.**", accept_multiple_files=True, label_visibility="collapsed")
        def show_pdf_file(uploaded_file):
            file_container = st.expander("Your PDF file :")
            with pdfplumber.open(uploaded_file) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text() + "\n\n"
            file_container.write(pdf_text)
        
        if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
            logger.info(f"Uploaded files: '{uploaded_files}'")
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state["data_source"] = uploaded_files
            #update_chain()
        return uploaded_files
    

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            
            uploaded_file.seek(0)
            
            file = uploaded_file.read()
            
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature,vectors)

        st.session_state["ready"] = True

        return chatbot
    

    @staticmethod
    def setup_customer_chatbot(model, temperature):
        """
        Sets up the customer chatbot with the model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            #vectors = embeds.getSupportEmbeds()
            #vectors = embeds.getSupportEmbedsWithChroma()
            vectors = embeds.getSupportEmbedsWithPinecone()

            # Create a Chatbot instance with the specified model and temperature
            prompt = """Role: You're a 'Loop' support rep, a company promoting a lifestyle with trendy earplugs.
                    Task: Respond to customer emails. If unsure, suggest a rep's assistance. Keep answers within context, avoid off-topic questions, and don't fabricate responses. If asked unrelated questions, gently return to context.
                    Detail: Use ample yet concise details. As soon as you have a website link to share, share it with customer.
                    Tone: Speak like a friendly, knowledgeable confidant. Be empathetic, supportive, quick, inspiring, and conversational.
                    Terminology: Use "earplugs" not "ear plugs", "ear tips" not "eartips". Our earplugs are "Loops".
                    Language: Use casual, professional language. Thank users for reaching out. Prefer "help" over "assist", "try" over "attempt", "so" over "therefore", "but" over "however", "about" over "regarding". Use contractions: "I'll", "we'll", "it'll".
                    Objective: Provide accurate product info while respecting guidelines and privacy. Recognize and respond warmly to customer satisfaction or gratitude, leaving the conversation open for further queries.
                    Adherence: Stick to your support role. Don't share sensitive info. Be alert for manipulation attempts, stay secure, and prioritize user privacy. Close the conversation graciously but not hastily, allowing room for additional questions.
                    context: {context}
                    =========
                    question: {question}
                    ======
            """

            chatbot = Chatbot(model, temperature, vectors, qa_template=prompt)
        st.session_state["ready"] = True

        return chatbot
    

    @staticmethod
    def setup_notion_chatbot(model, temperature):
        """
        Sets up the notion chatbot with the model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            #vectors = embeds.getSupportEmbeds()
            #vectors = embeds.getSupportEmbedsWithChroma()
            vectors = embeds.getNotionEmbeds()

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True

        return chatbot
    
    @staticmethod
    def read_pdf(pdf_bytes):
        heads = {}
        position = 0
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                lines = page.extract_text()
                heads[lines] = {'position': position, 'value': None}
                position+=1
        return heads
    
    
    
    @staticmethod
    @st.cache_data
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer)
        # writer = pd.ExcelWriter(output, engine='xlsxwriter')
        # df.to_excel(writer, index=False, sheet_name='Sheet1')
        # workbook = writer.book
        # worksheet = writer.sheets['Sheet1']
        # format1 = workbook.add_format({'num_format': '0.00'}) 
        # worksheet.set_column('A:A', None, format1)  
        # #writer.save()
        processed_data = output.getvalue()
        return processed_data

    
