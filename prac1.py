import streamlit as st
    
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS as TempFAISS
from langchain.text_splitter import CharacterTextSplitter
# Enhanced Styling – With Color Tweaks and UI Polish
st.markdown("""
    <style>
        *{
            
            margin: 0;
            padding: 0;

        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
        }
        /* Animated Gradient Sidebar with Soft Overlay */
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #ff5f6d, #ffc371, #7EE8FA, #EEC0C6, #FECF71);
            background-size: 600% 600%;
            animation: gradientFlow 15s ease infinite;
            color: white;
            border-radius: 24px;
            padding: 32px;
            font-weight: bold;
            box-shadow: 0 0 18px rgba(0,0,0,0.25);
        }

        .sidebar {
            border-radius: 12px;
        }

        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .sidebar .sidebar-header {
            font-size: 28px;
            font-weight: bold;
            color: white;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.5);
            margin-bottom: 24px;
        }

        .sidebar .sidebar-radio div {
            padding: 16px;
            font-size: 18px;
            border-radius: 14px;
            margin-bottom: 16px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
            text-align: center;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.25);
            background: rgba(255, 255, 255, 0.1);
        }

        .sidebar .sidebar-radio div:hover {
            background-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px) scale(1.03);
        }

        .chatbot-name {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 8px;
            display: inline-block;
        }

        .codebot-name { color: #4CAF50; }
        .chatmate-name { color: #FF6F61; }
        .docubot-name { color: #2196F3; }
        .ideaspark-name { color: #FF9800; }
        .techbot-name { color: #9C27B0; }

        .bot-response {
            padding: 20px;
            border-radius: 16px;
            margin: 20px 0;
            animation: fadeIn 0.8s ease forwards;
            font-size: 1.1em;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            line-height: 1.6;
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(12px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .codebot-response {
            background-color: #E6F4EA;
            color: #256029;
        }

        .chatmate-response {
            background-color: #FFEBEE;
            color: #C62828;
        }

        .docubot-response {
            background-color: #E3F2FD;
            color: #1565C0;
        }

        .ideaspark-response {
            background-color: #FFF3E0;
            color: #EF6C00;
        }

        .techbot-response {
            background-color: #F3E5F5;
            color: #6A1B9A;
        }

        /* Chat Input Section */
        .chat-input-container {
            display: flex;
            align-items: center;
            background: #ffffff;
            border-radius: 50px;
            padding: 12px 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-top: 32px;
        }

        .chat-input {
            flex: 1;
            border: none;
            outline: none;
            font-size: 1.1em;
            padding: 10px 14px;
            border-radius: 25px;
            background: #f5f5f5;
        }

        .send-button {
            width: 48px;
            height: 48px;
            border-radius: 80%;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
            margin-left: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        }

        .send-button:hover {
            transform: scale(1.1);
        }

        .send-icon {
            color: white;
            font-size: 20px;
        }

        .send-button-codebot { background-color: #4CAF50; }
        .send-button-chatmate { background-color: #FF6F61; }
        .send-button-docubot { background-color: #2196F3; }
        .send-button-ideaspark { background-color: #FF9800; }
        .send-button-techbot { background-color: #9C27B0; }

        /* Optional user message styling */
        .user-message {
            font-weight: 600;
            margin-top: 20px;
            color: #333;
        }
        .st-ak {
            gap: 12px;
        }
        .st-emotion-cache-6qob1r {
            position: relative;
            height: 100%;
            width: 100%;
            overflow: overlay;
            background:linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
            color:white;
        }

        .st-emotion-cache-p7i6r9 p {
            color:white;
            word-break: break-word;
            margin: 0px;
        }

        .st-cf {
            background-color: rgb(0 0 0);
        }

        .st-by {
    background-color: rgb(159 162 170);
}

        .st-av {
            background-color: white;
        }

        .st-ak {
            gap: 12px;
        }
        
   /* Subheader */
section[data-testid="stSidebar"] h3 {
    font-size: 25px;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
    margin-bottom: 10px;
}

/* Markdown italic text */
section[data-testid="stSidebar"] p {
    font-style: italic;
    font-size: 17px;
    font-weight: bold;
    text-align: center;
    color: white;
}

/* Radio buttons layout */
section[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 0 10px;
}

/* Radio input (circle) */
section[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #f59e0b;
    margin: 0;
}

/* Label styling */
section[data-testid="stSidebar"] label {
    display: flex;
    align-items: center;
    font-size: 16px !important;
    color: #000000;
    background-color: transparent;
    padding: 12px 111px 12px 12px;
    
    transition: all 0.3s ease;
    font-weight: bold;
    border-bottom: 1px solid #00000042;
    cursor: pointer;
}

/* Hover effect */
section[data-testid="stSidebar"] label:hover {
    background-color: rgba(255, 255, 255, 0.3);
    color: white;
}

/* Selected radio label */
section[data-testid="stSidebar"] input[type="radio"]:checked + label {
    background-color: #f59e0b;
    color: #111827;
    font-weight: bold;
}
.st-emotion-cache-zaw6nw {
        background: #e7c2c28a;  /* Gradient red background */
        color: white;  /* White text */
        border: none;  /* Remove border */
        padding: 10px 20px;  /* Button padding */
        font-size: 16px;  /* Font size */
        font-weight: bold;  /* Bold text */
        border-radius: 5px;  /* Rounded corners */
         /* Optional: Add shadow */
        cursor: pointer;  /* Pointer cursor on hover */
    }

    .st-emotion-cache-zaw6nw:hover {
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Reverse gradient on hover */
    }
    </style>
    
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN="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"    
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
        task="text-generation",
        max_new_tokens=512  # ✅ This goes here now
    )
    return llm

# Sidebar with bot selection
# Highlighted main bot at the top

# Initialize toggle state
if "show_all_bots" not in st.session_state:
    st.session_state.show_all_bots = False

# Toggle button
if st.sidebar.button("Switch to Other Bots"):
    st.session_state.show_all_bots = not st.session_state.show_all_bots

# Define bot options
if st.session_state.show_all_bots:
    options = [ "CodeBot", "ChatMate", "DocuBot", "IdeaSpark", "TechBot"]
else:
    options = ["TOMY"]  # If you want the second to be blank or placeholder

# Radio selection
tab = st.sidebar.radio("Choose your assistant", options)
if tab == "TOMY":
    st.markdown("""
        <span style='
            font-size: 0.9em;
            color: gray;
            background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            background-size: 100% 2px;
            background-repeat: no-repeat;
            background-position: 0 100%;
        '>
            You selected Tomy .
        </span>
    """, unsafe_allow_html=True)
elif tab == "---":
    st.markdown("""
        <span style='
            font-size: 0.9em;
            color: gray;
            background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            background-size: 100% 2px;
            background-repeat: no-repeat;
            background-position: 0 100%;
        '>
            Please select a valid bot.
        </span>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <span style='
            font-size: 0.9em;
            color: gray;
            background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            background-size: 100% 2px;
            background-repeat: no-repeat;
            background-position: 0 100%;
        '>
            You selected {tab}.
        </span>
    """, unsafe_allow_html=True)

def get_sidebar_css(bot_name):
    if bot_name == "TOMY":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                       background:black !important /* Dark gradient for the background */
                    color: black;  /* White text */
                    border-right: 4px solid;
                    border-image: linear-gradient(to bottom, red, orange, yellow, green, blue, indigo, violet);
                    border-image-slice: 1;
                    
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
                }
                section[data-testid="stSidebar"] label {
                        
                        border-bottom: 1px solid;
                        background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            background-size: 100% 1px;
            background-repeat: no-repeat;
            background-position: 0 100%;
                }
                .st-emotion-cache-bho8sy {
                        background-color: #5e808d69;  /* Solid black for message input background */
                    color: black;  /* White text for message input */
                }
                .st-chat-input input {
                    color: black;  /* White input text */
                    background-color: #333333;  /* Dark gray for input box */
                }
                .st-chat-message {
                    color: black;  /* White text for chat messages */
                }
                .stButton>button {
                    background-color: #333333;  /* Dark button background */
                      /* White button text */
                      /* Subtle border for button */
                     border: 1px solid;
            border-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            border-image-slice: 1;
            padding: 2px 6px;
            border-radius: 4px;
                }
                .stButton>button:hover {
                    background-color: #666666;  /* Darker button on hover */
                }
                .stMarkdown {
                    color: black;  /* White text for markdown content */
                }
            </style>
        """
    elif bot_name == "CodeBot":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                    background: linear-gradient(135deg, #4CAF50, #388E3C);
                    color: white;
                }
                .st-emotion-cache-bho8sy {
                    background-color: rgb(76 175 80);
                }
            </style>
        """
    elif bot_name == "ChatMate":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                    background: linear-gradient(135deg, #FF6F61, #FF4081);
                    color: white;
                }
                .st-emotion-cache-bho8sy {
                    background-color: rgb(255 111 97);
                }
            </style>
        """
    elif bot_name == "DocuBot":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                    background: linear-gradient(135deg, #2196F3, #1976D2);
                    color: white;
                }
                .st-emotion-cache-bho8sy {
                    background-color: rgb(33 158 246);
                }
            </style>
        """
    elif bot_name == "IdeaSpark":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                    background: linear-gradient(135deg, #FF9800, #FF5722);
                    color: white;
                }
                .st-emotion-cache-bho8sy {
                    background-color: rgb(255 152 0);
                }
            </style>
        """
    elif bot_name == "TechBot":
        return """
            <style>
                .st-emotion-cache-6qob1r {
                    background: linear-gradient(135deg, #9C27B0, #8E24AA);
                    color: white;
                }
                .st-emotion-cache-bho8sy {
                    background-color: rgb(156 39 176);
                }
            </style>
        """
    return ""

# Inject the CSS based on selected bot
st.markdown(get_sidebar_css(tab), unsafe_allow_html=True)

# If BotGuru is selected, show its content first
if tab == "TOMY":
    st.markdown("<span class='chatbot-name botguru-name'>Tomy</span>", unsafe_allow_html=True)
    st.markdown("I'm a bot trained to answer based on the trusted information my creator provided — mainly right now from the **Gale Encyclopedia of Medicine**. You can also upload your own PDF!")

    # Upload Option
    uploaded_file = st.file_uploader("📄 Upload a PDF to chat with instead", type="pdf")

    # Function to process PDF and create temporary vectorstore
    def process_pdf(file):
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader("temp_uploaded.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = TempFAISS.from_documents(docs, embedding_model)
        return db

    def main():
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt = st.chat_input("Ask your question here...")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Use PDF if available, otherwise Gale
            if uploaded_file:
                vectorstore = process_pdf(uploaded_file)
                st.success("Tomy is now answering based on your uploaded PDF.")
            else:
                vectorstore = get_vectorstore()
                st.info("Tomy is answering based on the Gale Encyclopedia of Medicine.")

            CUSTOM_PROMPT_TEMPLATE = """
            You're a helpful and friendly assistant. Your name is Tomy. Use the information provided in the context below to answer the user's question as clearly and kindly as possible.

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
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
                )

                # Friendly greetings shortcut
                greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
                if any(greet in prompt.lower() for greet in greetings):
                    result = "Hello! 😊 How can I help you today?"
                else:
                    response = qa_chain.invoke({"query": prompt})
                    result = response["result"]

                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

                # Optional: show sources
                if not any(greet in prompt.lower() for greet in greetings):
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(response["source_documents"], start=1):
                            st.markdown(f"**Source {i}:**\n{doc.page_content}")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

    if __name__ == "__main__":
        main()

# Handle the other bots
elif tab == "CodeBot":
    st.markdown("<span class='chatbot-name codebot-name'>CodeBot</span>", unsafe_allow_html=True)
    st.markdown("You're now chatting with **CodeBot**. It’s great for writing, debugging, and explaining code.")
    if 'message5' not in st.session_state:
        st.session_state.message5 = []
    for message in st.session_state.message5:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask anything here")
    if prompt:
        st.chat_message("User").markdown(prompt)
        st.session_state.message5.append({'role': 'User', 'content': prompt})
        response = "Hi I am Shayan"
        st.chat_message("assistant").markdown(response)
        st.session_state.message5.append({'role': 'assistant', 'content': response})
    
elif tab == "ChatMate":
    st.markdown("<span class='chatbot-name chatmate-name'>ChatMate</span>", unsafe_allow_html=True)
    st.markdown("You're now chatting with **ChatMate**. A casual, human-like conversationalist.")
    if 'message1' not in st.session_state:
        st.session_state.message1 = []
    for message in st.session_state.message1:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask anything here")
    if prompt:
        st.chat_message("User").markdown(prompt)
        st.session_state.message1.append({'role': 'User', 'content': prompt})
        response = "Hi I am Shayan"
        st.chat_message("assistant").markdown(response)
        st.session_state.message1.append({'role': 'assistant', 'content': response})

elif tab == "DocuBot":
    st.markdown("<span class='chatbot-name docubot-name'>DocuBot</span>", unsafe_allow_html=True)
    st.markdown("You're now chatting with **DocuBot**. It summarizes and rewrites documents.")
    if 'message2' not in st.session_state:
        st.session_state.message2 = []
    for message in st.session_state.message2:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask anything here")
    if prompt:
        st.chat_message("User").markdown(prompt)
        st.session_state.message2.append({'role': 'User', 'content': prompt})
        response = "Hi I am Shayan"
        st.chat_message("assistant").markdown(response)
        st.session_state.message2.append({'role': 'assistant', 'content': response})

elif tab == "IdeaSpark":
    st.markdown("<span class='chatbot-name ideaspark-name'>IdeaSpark</span>", unsafe_allow_html=True)
    st.markdown("You're now chatting with **IdeaSpark**. Perfect for creative ideas and brainstorming.")
    if 'message3' not in st.session_state:
        st.session_state.message3 = []
    for message in st.session_state.message3:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask anything here")
    if prompt:
        st.chat_message("User").markdown(prompt)
        st.session_state.message3.append({'role': 'User', 'content': prompt})
        response = "Hi I am Shayan"
        st.chat_message("assistant").markdown(response)
        st.session_state.message3.append({'role': 'assistant', 'content': response})

elif tab == "TechBot":
    st.markdown("<span class='chatbot-name techbot-name'>TechBot</span>", unsafe_allow_html=True)
    st.markdown("You're now chatting with **TechBot**. It helps with tech questions and troubleshooting.")
    if 'message4' not in st.session_state:
        st.session_state.message4 = []
    for message in st.session_state.message4:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask anything here")
    if prompt:
        st.chat_message("User").markdown(prompt)
        st.session_state.message4.append({'role': 'User', 'content': prompt})
        response = "Hi I am Shayan"
        st.chat_message("assistant").markdown(response)
        st.session_state.message4.append({'role': 'assistant', 'content': response})
