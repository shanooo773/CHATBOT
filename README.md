# 🤖 CHATBOT - Multi-Purpose AI Assistant Platform

A sophisticated chatbot application built with Python, featuring multiple specialized AI assistants powered by advanced language models and retrieval-augmented generation (RAG) technology.

## 🌟 Features

### 🎯 Multiple Specialized Chatbots
- **TOMY** - Main assistant trained on medical encyclopedia data with PDF upload capability
- **CodeBot** - Programming and code debugging specialist
- **ChatMate** - Casual, human-like conversationalist
- **DocuBot** - Document summarization and rewriting expert
- **IdeaSpark** - Creative ideas and brainstorming assistant
- **TechBot** - Technical support and troubleshooting helper

### 🚀 Advanced Capabilities
- **RAG (Retrieval Augmented Generation)** - Combines document knowledge with LLM responses
- **PDF Document Processing** - Upload and chat with your own PDF documents
- **Vector Database Search** - FAISS-powered similarity search for relevant information
- **Pre-trained Knowledge Base** - Built-in Gale Encyclopedia of Medicine
- **Responsive Web Interface** - Beautiful Streamlit-based UI with custom styling
- **Real-time Chat** - Interactive conversation with persistent message history

## 🛠️ Technologies Used

- **Python 3.12** - Core programming language
- **Streamlit** - Web application framework
- **LangChain** - LLM orchestration and RAG implementation
- **HuggingFace Transformers** - Language models and embeddings
- **FAISS** - Vector similarity search and storage
- **PyPDF** - PDF document processing
- **Sentence Transformers** - Text embeddings generation

### 🤖 AI Models
- **Mistral-7B-Instruct-v0.3** - Primary language model via HuggingFace
- **all-MiniLM-L6-v2** - Sentence embedding model for document retrieval

## 📋 Prerequisites

- Python 3.12 or higher
- HuggingFace API token (for model access)
- Minimum 8GB RAM recommended
- Internet connection for model downloads

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shanooo773/CHATBOT.git
cd CHATBOT
```

### 2. Install Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using Pipenv**
```bash
pipenv install
pipenv shell
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_api_token_here
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### 4. Initialize Vector Database
Run the memory setup to process the included medical encyclopedia:
```bash
python memory.py
```

## 💻 Usage

### Starting the Application

**Main Web Interface:**
```bash
streamlit run prac1.py
```

**Alternative Interface:**
```bash
streamlit run web.py
```

**Command Line Interface:**
```bash
python llm.py
```

### 🌐 Web Interface Usage

1. **Open your browser** to `http://localhost:8501`
2. **Select a chatbot** from the sidebar:
   - Toggle between TOMY and other specialized bots
3. **Start chatting** by typing in the input field
4. **Upload PDFs** (TOMY only) to chat with custom documents
5. **View source documents** using the expandable section

### 📄 PDF Upload Feature

1. Select **TOMY** from the sidebar
2. Click **"Upload a PDF to chat with instead"**
3. Choose your PDF file
4. Start asking questions about the uploaded document
5. TOMY will answer based on your PDF content instead of the default encyclopedia

## 📁 Project Structure

```
CHATBOT/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── Pipfile               # Pipenv configuration
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
│
├── web.py               # Alternative Streamlit interface
├── prac1.py             # Main Streamlit application
├── llm.py               # Core LLM functionality
├── memory.py            # Document processing and vectorstore creation
│
├── data/                # Document storage
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
│
├── vectorstore/         # FAISS vector database
│   └── db_faiss/       # Processed embeddings
│
└── temp_uploaded.pdf    # Temporary file for uploaded PDFs
```

## 🔧 Configuration

### Model Configuration
- **Primary LLM:** Mistral-7B-Instruct-v0.3
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Temperature:** 0.9 (adjustable in code)
- **Max Tokens:** 512
- **Retrieval Documents:** 3 per query

### FAISS Database
- **Storage Path:** `vectorstore/db_faiss`
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 50 characters

## 🎨 Customization

### Adding New Documents
1. Place PDF files in the `data/` directory
2. Run `python memory.py` to reprocess the vectorstore
3. The new documents will be available for querying

### Modifying Chatbot Behavior
- Edit prompt templates in `llm.py` or `prac1.py`
- Adjust model parameters (temperature, max_tokens)
- Customize the retrieval system settings

### UI Customization
- Modify CSS styles in `prac1.py`
- Add new chatbot personalities
- Customize color schemes and layouts

## 🚨 Troubleshooting

### Common Issues

**1. HuggingFace Token Error**
```
Error: Authentication failed
```
- Verify your HF_TOKEN in the `.env` file
- Ensure the token has appropriate permissions

**2. FAISS Database Not Found**
```
Error: vectorstore/db_faiss not found
```
- Run `python memory.py` to create the database
- Ensure PDF files exist in the `data/` directory

**3. Memory Issues**
```
CUDA out of memory / RAM error
```
- Use CPU-only mode by setting `device='cpu'`
- Reduce chunk size or batch size
- Close other applications to free memory

**4. Streamlit Port Conflicts**
```
Port 8501 is already in use
```
- Use a different port: `streamlit run prac1.py --server.port 8502`
- Kill existing Streamlit processes

## 🔒 Security Notes

- Keep your HuggingFace API token secure
- Never commit `.env` files to version control
- Be cautious when uploading sensitive PDFs
- The application processes uploaded files locally

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/new-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes:** `git commit -m 'Add new feature'`
5. **Push to the branch:** `git push origin feature/new-feature`
6. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 Python style guidelines
- Add docstrings to new functions
- Test changes with different PDF types
- Update documentation for new features

## 📜 License

This project is open source. Please ensure compliance with:
- HuggingFace model licenses
- Third-party library licenses
- PDF content usage rights

## 🙏 Acknowledgments

- **HuggingFace** for providing the language models
- **LangChain** for the RAG framework
- **Streamlit** for the web interface
- **Meta AI** for FAISS vector search
- **Gale Encyclopedia** for the medical knowledge base

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the documentation and code comments

---

**Happy Chatting! 🎉**