import streamlit as st
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from src.backend.prompts import REPORT_PROMPT, SEPARATOR_PROMPT
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

#fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, 
                model_name="gpt-3.5-turbo", 
                temperature=0.7, 
                vectors=None,
                qa_template = """Use the following pieces of context to answer the question at the end, use as much informations you have to make your answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                                {context}

                                Question: {question}
                                Helpful Answer:"""
                    ):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})
        


        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        result["answer"] = clean_answer(result["answer"])

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]
    
    def report_writter(self, job_descr, profile):
        """
        Start a reports producer with a model via Langchain
        """
        
        prompt = REPORT_PROMPT.format(job_description=job_descr, resume=profile)
        llm = OpenAI(model_name="gpt-4-1106-preview", temperature=0.2)
        result = llm(prompt)

        return result
    
    def resume_analysis(self, text):
        """
        Start a resume analyser, say if a text is the beginning or the continuation of a CV
        """
        chat = ChatOpenAI(model_name="gpt-4", temperature=0.2)
        messages = [
            SystemMessage(content=SEPARATOR_PROMPT),
            HumanMessage(content=text)
        ]
        result = chat(messages)
        return result.content
    



def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 


def predict_amount_token(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

    
def clean_answer(answer):
        prefixes = ["Answer:", "Response:"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                return answer[len(prefix):].strip()
        else:
            return answer
