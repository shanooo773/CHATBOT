import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = 
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.9,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512},
        task="text-generation"
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your question here...")  # ✅ This was previously outside

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        You're a helpful and friendly assistant. Use the information provided in the context below to answer the user's question as clearly and kindly as possible.

        If you're not sure about the answer, it's okay to say you don't know—please don't guess or add anything beyond what's given.

        Be concise, helpful, and respectful.

        ---
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            # Handle greetings separately
            greetings = ["hi", "hello", "hey", "how are you", "how are you?", "good morning", "good evening"]
            normalized_prompt = prompt.lower().strip()

            if any(greet in normalized_prompt for greet in greetings):
                result = "Hello! 😊 How can I help you today?"
                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
            else:
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]
                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(response["source_documents"], start=1):
                        st.markdown(f"**Source {i}:**\n{doc.page_content}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
