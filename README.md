# 🤖 RAG Document Assistant

A simple, intelligent document Q&A system built with Streamlit, Pinecone, and LangChain. Ask questions about your documents and get relevant answers with source citations.

## ✨ Features

- **Document Processing**: Load PDF and TXT files from local folders
- **Website Scraping**: Parse XML sitemaps to extract web content
- **Smart Search**: Use semantic search to find relevant document chunks
- **AI-Powered Answers**: Get concise answers based on retrieved context
- **Source Citations**: Always see where your answers come from
- **No External LLM**: Works offline with simple snippet-based responses

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Pinecone API key
- Hugging Face API key (for embeddings only)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd botrag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 How to Use

### 1. Setup API Keys
- Enter your **Pinecone API key** in the sidebar
- Enter your **Hugging Face API key** in the sidebar

### 2. Load Documents
Choose your data source:

**Option A: Local Files**
- Select "Local folder" as data source
- Put your PDF/TXT files in the `data/` folder
- Click "Load data to Pinecone"

**Option B: Website Sitemap**
- Select "Sitemap URL" as data source
- Update `WEBSITE_URL` in `constants.py` if needed
- Click "Load data to Pinecone"

### 3. Ask Questions
- Type your question in the text input
- Adjust the number of document chunks to retrieve (0-5)
- Click "Search" to get your answer

## 🏗️ Project Structure

```
botrag/
├── app.py              # Main Streamlit application
├── utils.py            # Core RAG functions
├── constants.py        # Configuration settings
├── requirements.txt    # Python dependencies
├── data/              # Put your documents here
└── README.md          # This file
```

## ⚙️ Configuration

Edit `constants.py` to customize:
- `WEBSITE_URL`: Sitemap URL for web scraping
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX`: Your Pinecone index name

## 🔧 How It Works

1. **Document Loading**: PDFs/TXTs are loaded and split into chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone vector database
4. **Query Processing**: Your question is converted to embeddings
5. **Similarity Search**: Most relevant document chunks are retrieved
6. **Answer Generation**: Relevant snippets are extracted and displayed
7. **Source Citation**: Original document sources are shown

## 📚 Supported File Types

- **PDF**: `.pdf` files
- **Text**: `.txt` files
- **Websites**: XML sitemaps

## 🛠️ Dependencies

- `streamlit`: Web interface
- `pinecone`: Vector database
- `langchain`: Document processing and embeddings
- `sentence-transformers`: Text embeddings
- `pypdf`: PDF parsing
- `bs4`: Web scraping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

**Common Issues:**

- **Pinecone Connection Error**: Check your API key and environment settings
- **No Documents Found**: Ensure you've loaded data before searching
- **Import Errors**: Reinstall dependencies with `pip install -r requirements.txt`

**Need Help?**
- Check the error messages in the Streamlit interface
- Verify your API keys are correct
- Ensure your Pinecone index exists and has the right dimensions (384 for all-MiniLM-L6-v2)

## 🎯 Future Enhancements

- [ ] Add support for more file types (DOCX, CSV, etc.)
- [ ] Implement conversation memory
- [ ] Add document upload interface
- [ ] Support for multiple Pinecone indexes
- [ ] Export search results
- [ ] User authentication

---

**Built with ❤️ using Streamlit, Pinecone, and LangChain**
