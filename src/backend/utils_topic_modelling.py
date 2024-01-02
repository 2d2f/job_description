import nltk
nltk.download('punkt')

import pandas as pd
import json
import asyncio
import streamlit as st

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize

from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from src.backend.prompts import TOPIC_PROMPT, CLASSIFY_PROMPT, CLASS_PROMPT

from typing import List, Dict, Any
import pandas as pd

from langchain_core.runnables.base import RunnableSerializable

@st.cache_data
def get_representative_dataset(df: pd.DataFrame) -> List[str]:
    """
    Extracts representative documents from a dataset using topic modeling.

    This function processes a dataframe containing text data, applies a BERT-based topic modeling approach using 
    the BERTopic library, and returns a collection of documents that are representative of each identified topic.

    :param df: A pandas DataFrame with a column named "Review Content" containing text documents.
    :return: A list of representative documents for each topic identified in the dataset.

    The function uses a KeyBERT-inspired model for representation and a CountVectorizer for vectorization, with
    a minimum document frequency (min_df) of 2. It fits a BERTopic model to the text data, identifies topics, 
    and then retrieves representative documents for each topic. Finally, it aggregates these documents and 
    returns them as a list of strings.
    """

    docs = list(df["Review Content"])

    representation_model = KeyBERTInspired()

    # N.B. : min_df : The min_df parameter is set to 5, which means it will ignore terms that appear in fewer than 5 documents. If your dataset is small or if the distribution of terms is very sparse, this threshold might be too high. Try reducing min_df to a lower value, such as 2 or 3, or even a float representing a proportion of documents (e.g., 0.01 for 1%).
    vectorizer_model = CountVectorizer(min_df=3, stop_words = 'english')

    topic_model = BERTopic(nr_topics = 'auto', vectorizer_model = vectorizer_model,
                        representation_model = representation_model)

    # We fit the model on our dataset
    topics, ini_probs = topic_model.fit_transform(docs)

    # Getting representative documents for each topic
    representative_docs = topic_model.get_representative_docs()

    # Creating a DataFrame
    topic_stats_df = pd.DataFrame(representative_docs.items(), columns=['Topic', 'Representative_Docs'])

    repr_docs = topic_stats_df.Representative_Docs.sum()

    return repr_docs


# TODO : faire attention à la taille du prompt, diviser le prompt au besoin (ne pas envoyer tous les reviews d'un coup)
@st.cache_data
def get_topics_list(repr_docs: List[str]) -> List[Dict]:
    """
    Processes a list of representative documents using an AI model to extract topics.

    This function sends a concatenated string of representative documents to an AI model (specifically, OpenAI's GPT-3.5-turbo model)
    to extract topics from these documents. It returns a list of dictionaries, each representing a topic extracted from the input documents.

    :param repr_docs: A list of strings, where each string is a representative document from a dataset.
    :return: A list of dictionaries, where each dictionary contains details about a topic extracted from the representative documents.

    The function concatenates the representative documents using a specified delimiter and sends this concatenated string to the AI model.
    The response is parsed as JSON to extract the topics, which are then returned as a list of dictionaries.
    """
    
    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

    delimiter = '####'
    input = delimiter.join(repr_docs)

    chain = TOPIC_PROMPT | model

    topics_response = chain.invoke({"delimiter": delimiter, "reviews":input})

    topics_list = json.loads(topics_response.content)

    return topics_list

# Need to further analyse caching in streamlit applications for chain of langchain library (got UnhashableParamError)
def hash_runnable_serializable(obj: RunnableSerializable) -> int:
    # Extract key attributes
    key_attrs = [getattr(obj, attr, None) for attr in obj.__fields__.keys()]

    # Create a unique string representation
    unique_representation = ''.join(str(attr) for attr in key_attrs)

    # Return the hash of this unique representation
    return hash(unique_representation)

# Use this function in st.cache_data
#@st.cache_data(hash_funcs={RunnableSerializable: hash_runnable_serializable})
async def classify_review(chain: RunnableSerializable, delimiter: str, topics_descr_list_str: str, title: str, content: str) -> str:
    """
    Asynchronously classifies a review based on the provided title and content.

    This function sends a formatted string containing the review's title and content to an AI model chain for classification. 
    It awaits the response and then returns the content of that response.

    :param chain: The AI model chain to be used for classification.
    :param delimiter: A string used as a delimiter in the AI model's input.
    :param topics_descr_list_str: A string that contains the concatenated topic names and descriptions.
    :param title: The title of the review.
    :param content: The content of the review.
    :return: The content of the response from the AI model, as a string.

    The function constructs the input by concatenating the title and content of the review and sends it to the AI model.
    """
    input = "Title : " + title + "\nContent :" + content
    attempts = 0
    max_retries = 3

    while attempts < max_retries:
        response = await chain.ainvoke({"delimiter": delimiter, "topics_list_str": topics_descr_list_str, "customer_review": input})
        cleaned_response = response.content.replace('```json\n', '').replace('```', '').strip()

        try:
            response_json = json.loads(cleaned_response)
            normalized_response = normalize_sentiments(response_json)
            return normalized_response
        except json.JSONDecodeError:
            # Increment the attempt counter
            attempts += 1
            if attempts >= max_retries:
                # Handle the maximum retry limit reached scenario
                return cleaned_response  # or any other appropriate action

    # If the loop exits without a return, it means max retries were reached
    return cleaned_response  # or handle appropriately


#TODO : review batches of several reviews in one shot
async def classify_reviews(df: pd.DataFrame, topics_list: List[Dict]) -> pd.DataFrame:
    """
    Asynchronously classifies each review in a DataFrame using an AI model.

    This function iterates over each row in the DataFrame, classifies the review based on its title and content using an AI model, 
    and adds the classification results as a new column in the DataFrame.

    :param df: A pandas DataFrame with columns "Review Title" and "Review Content".
    :param topics_list: A list of dictionaries, each containing 'topic_name' and 'topic_description'.
    :return: The original DataFrame with an additional column 'Topics' containing the classification results.

    The function prepares asynchronous tasks for each row in the DataFrame and then executes them concurrently.
    The responses are aggregated and added as a new column in the DataFrame.
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

    topics_descr_list_str = '\n'.join(map(lambda x: x['topic_name'] + ': ' + x['topic_description'], topics_list))
    delimiter = '###'
    chain = CLASS_PROMPT | model

    # Prepare and gather all async tasks
    tasks = []
    for index, row in df.iterrows():
        task = asyncio.create_task(classify_review(chain, delimiter, topics_descr_list_str, row["Review Title"], row["Review Content"]))
        tasks.append(task)

    # Execute all tasks asynchronously and retrieve responses
    responses = await asyncio.gather(*tasks)

    # Parse and expand the JSON responses into a list of dictionaries
    expanded_responses = []
    for index, response_json in enumerate(responses):
        # Check if the cleaned response is not empty
        if response_json:
            try:
                response_list = json.loads(response_json)  # Attempt to parse cleaned JSON
                for topic_info in response_list:
                    expanded_responses.append({
                        'Review Index': index,
                        'Topic': topic_info['topic'],
                        'Sentiment': topic_info['sentiment']
                    })
            except json.JSONDecodeError:
                # Handle invalid JSON
                print(f"Invalid JSON response for review at index {index}: {response_json}")
                # Optionally, add a placeholder or skip
                expanded_responses.append({
                    'Review Index': index,
                    'Topic': 'Invalid Response',
                    'Sentiment': None
                })
        else:
            # Handle empty response string
            expanded_responses.append({
                'Review Index': index,
                'Topic': 'No Response',
                'Sentiment': None
            })

    # Create a flat DataFrame from the expanded responses
    topics_df = pd.DataFrame(expanded_responses)

    # Merge this with the original DataFrame
    merged_df = df.reset_index().merge(topics_df, left_on='index', right_on='Review Index')
    merged_df.drop(['Review Index', 'index'], axis=1, inplace=True)

    return merged_df

@st.cache_data
def cached_classify_reviews(df: pd.DataFrame, topics_list: List[Dict]):
    df = asyncio.run(classify_reviews(df, topics_list))
    return df


def normalize_sentiments(response_json):
    normalized_response = []
    for item in response_json:
        # Normalize the sentiment to lowercase
        item['sentiment'] = item['sentiment'].lower()
        normalized_response.append(item)
    return json.dumps(normalized_response)


@st.cache_data
def aggregate_sentences_by_assigned_topics(df, topics_list, threshold=0.5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    aggregated_sentences = []

    # Convert topics list to a dictionary for easier access
    topic_descriptions = {topic['topic_name']: topic['topic_description'] for topic in topics_list}

    for _, row in df.iterrows():
        assigned_topic = row['Topic']
        review_sentences = sent_tokenize(row['Review Content'])

        # Get the description for the assigned topic
        assigned_description = topic_descriptions.get(assigned_topic, '')

        for sentence in review_sentences:
            # Calculate semantic similarity only for the assigned topic
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            description_embedding = model.encode(assigned_description, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(sentence_embedding, description_embedding).item()

            if similarity_score > threshold:
                aggregated_sentences.append({
                    'Sentence': sentence,
                    'Topic': assigned_topic,
                    'Sentiment': row['Sentiment'],
                    'Original Review': row['Review Content']
                })

    return pd.DataFrame(aggregated_sentences)
