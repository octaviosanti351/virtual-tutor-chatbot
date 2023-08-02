import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

import langchain
langchain.verbose = False
DEFAULT_VECTOR_STORE_PATH = os.getenv("DATABASE_DIRECTORY")

class Chatbot:

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    qa_template = """
            You are a helpful AI assistant named Robby. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        =========
        question: {question}
        ======
    """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature, openai_api_key=os.getenv("OPENAI_API_KEY"))

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        doc_search = FAISS.load_local(folder_path=DEFAULT_VECTOR_STORE_PATH, index_name="canvas-bot", embeddings=embeddings)
        retriever = doc_search.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
