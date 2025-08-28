# AI-RAG-Assistant-For-Website-Gemini-
## 🤖 Gemini Website RAG Assistant

An end‑to‑end Retrieval‑Augmented Generation (RAG) assistant that indexes website content (via sitemap) and local documents, stores embeddings in Pinecone, and answers questions using Google Gemini. Built with Streamlit and LangChain.

### ✨ Highlights
- **True RAG pipeline**: retrieval → augmentation → generation
- **Gemini‑powered** answers with cited sources
- **Hybrid ingestion**: website sitemap + local files (`documents/`)
- **Vector search** with Pinecone
- **Streamlit UI** with progress states and context viewer

---

## 🚀 Quick Start

### 1) Clone and install
```bash
pip install -r requirements.txt
```

### 2) Configure environment
Copy the template and add your keys:
```bash
copy env_example.txt .env
```
Required variables in `.env`:
```bash
GOOGLE_API_KEY=...
HUGGINGFACE_API_KEY=...
PINECONE_API_KEY=...
# Optional overrides
GEMINI_MODEL=gemini-1.5-flash
TEMPERATURE=0.7
MAX_TOKENS=1000
```

### 3) (Optional) Add local docs
Place `.txt`, `.pdf`, `.docx`, `.md` in `documents/` (subfolders allowed).

### 4) Run
```bash
streamlit run app.py
```
Enter your API keys in the sidebar → click “Load data to Pinecone” → ask questions.

---

## 🧱 Architecture
- **UI**: Streamlit (`app.py`)
- **RAG Core**: LangChain utilities (`utils.py`)
- **Vectors**: Pinecone (`langchain-pinecone`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **LLM**: Google Gemini (`langchain-google-genai`)
- **Ingestion**:
  - Website via sitemap (`SitemapLoader`)
  - Local files via loaders (TXT, PDF, DOCX, MD)

Data flow:
1. Load website + local docs → split into chunks
2. Create embeddings → upsert to Pinecone
3. Query: retrieve top‑k similar chunks
4. Augment prompt with context → generate answer with Gemini
5. Display answer + cited sources + expandable context

---

## ⚙️ Configuration
Adjust in `constants.py`:
```python
WEBSITE_URL = "https://www.indiancounsellingservices.com/job-listing/sitemap-1.xml"
DOCUMENT_SOURCES = {
    "website_url": WEBSITE_URL,
    "website_limit": 5,
    "local_path": "./documents",
    "file_types": [".txt", ".pdf", ".docx", ".md"]
}
```
Recommended Gemini models: `gemini-1.5-flash` (fast, cost‑effective) or `gemini-1.5-pro` (higher quality).

---

## 🧪 Try These Questions
- “What are the official working hours?”
- “How many sick leave days do employees get per year?”
- “How do I set up the SmartHome Hub for the first time?”
- “List all automation features supported by the hub.”
- “Summarize the onboarding journey from day 1 to 90 days.”

More prompts in `documents/test_scenarios.txt`.

---

## 📁 Project Structure
```
Ragbot for Website/
├─ app.py                # Streamlit app (Gemini UI)
├─ utils.py              # RAG pipeline, loaders, LLM, Pinecone
├─ constants.py          # URLs, index names, defaults
├─ requirements.txt      # Dependencies
├─ documents/            # Local docs to ingest
│  ├─ company_policies.txt
│  ├─ product_manual.txt
│  ├─ faq.txt
│  └─ training_materials.txt
└─ env_example.txt       # .env template
```

---

## 🧰 Troubleshooting
- Missing loaders (PDF/DOCX/MD)? Install:
```bash
pip install -U langchain-community pypdf docx2txt unstructured python-docx lxml bs4
```
- Gemini 404/model error → set a supported model in `.env`:
```bash
GEMINI_MODEL=gemini-1.5-flash
```
- Python 3.13 issues? Prefer 3.10/3.11 virtualenv:
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Pinecone index
  - Create index name from `constants.py` (default `ragbot`) in region `us-east-1`.

---

## 🔒 Notes
- The app shows sources for transparency; always verify critical answers.
- Respect robots.txt and target site policies when crawling sitemaps.
- Set a `USER_AGENT` env var if needed for polite crawling.

---

## 🗺️ Roadmap
- Chat history and memory
- Reranking for better retrieval quality
- Streaming responses
- Multi‑tenant/project profiles
- Evaluations (RAGAS) and observability

---

## 📝 License
MIT — feel free to use and modify. Attribution appreciated.

