import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint  # Updated import according to deprecation warnings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import according to deprecation warnings
from langchain_community.vectorstores import FAISS

# Load .env and get Hugging Face API token
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# Repo ID for Mistral model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Load the Language Model
def load_llm(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.9,
        huggingfacehub_api_token=HF_TOKEN,  # Correctly pass token
        model_kwargs={"max_length": 512},
        task="text-generation"  # Ensure the task is set to text-generation
    )
    return llm

# Step 2: Create a custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Do not try to make up an answer.
Do not provide anything outside of the given context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS DB and connect it with LLM
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Step 4: Accept a user query and get the result
if __name__ == "__main__":
    user_query = input("Write your question: ")
    response = qa_chain.invoke({"query": user_query})

    print("\n🔍 RESULT:\n", response["result"])
    print("\n📚 SOURCE DOCUMENTS:\n")
    for i, doc in enumerate(response["source_documents"], start=1):
        print(f"--- Source {i} ---")
        print(doc.page_content)
        print()
