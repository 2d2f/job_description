import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os


def handle_upload_files():
    data_source_container = st.container()
    # container to display infos stored to session state
    # as it needs to be accessed from submodules
    uploaded_files = data_source_container.file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'], label_visibility="collapsed")
    
    if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["data_source"] = uploaded_files
        #update_chain()
    return uploaded_files


# Function to read the uploaded file into a DataFrame
@st.cache_data
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            # Check the file extension and use the appropriate Pandas function
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
            return df
        except Exception as e:
            # Handle error in reading file
            st.error(f"Error reading file: {e}")
            return None
    else:
        # Handle case where no file was uploaded
        st.error("No file uploaded.")
        return None

@st.cache_data
def concatenate_reviews(df):
    df.dropna(subset=['Review Content'], inplace=True)
    concatenated_df = pd.DataFrame(columns=df.columns)
    concatenated_review = ""
    previous_title = None
    previous_score = None
    
    temp_data = []
    
    for index, row in df.iterrows():
        if pd.notnull(row['Review Title']):
            if concatenated_review:
                temp_data.append({
                    'Review Content': concatenated_review.strip(),
                    'Review Title': previous_title,
                    'Review Score': previous_score
                })
                concatenated_review = ""
            concatenated_review += str(row['Review Content'])
            previous_title = row['Review Title']
            previous_score = row['Review Score']
        else:
            concatenated_review += " " + str(row['Review Content'])
    
    if concatenated_review:
        temp_data.append({
            'Review Content': concatenated_review.strip(),
            'Review Title': previous_title,
            'Review Score': previous_score
        })
    
    concatenated_df = pd.DataFrame(temp_data)
    return concatenated_df

@st.cache_data
def first_insights(df):
    # Categorizing the reviews based on their score
    likes = df[df['Review Score'] >= 4]
    dislikes = df[df['Review Score'] <= 2]
    neutral = df[df['Review Score'] == 3]

    # Counting the total number of reviews in each category
    total_reviews = len(df)
    num_likes = len(likes)
    num_dislikes = len(dislikes)
    num_neutral = len(neutral)

    # Calculating the percentages
    percent_likes = (num_likes / total_reviews) * 100
    percent_dislikes = (num_dislikes / total_reviews) * 100
    percent_neutral = (num_neutral / total_reviews) * 100

    # Identifying explicit complaints in the review content, regardless of score
    # complaints = data[data['Review Content'].str.contains("poor|bad|problem|issue|disappoint|not good|worse|worst", case=False, na=False)]
    # num_complaints = len(complaints)
    # percent_complaints = (num_complaints / total_reviews) * 100

    return percent_likes, percent_dislikes, percent_neutral#, percent_complaints, num_complaints


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