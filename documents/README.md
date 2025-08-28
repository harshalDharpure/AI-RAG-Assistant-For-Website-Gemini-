# 📁 Documents Folder

This folder contains documents that will be automatically loaded into the RAG system.

## 📋 **Supported File Types**

- **`.txt`** - Plain text files
- **`.pdf`** - PDF documents  
- **`.docx`** - Microsoft Word documents
- **`.md`** - Markdown files

## 🚀 **How to Use**

1. **Place your documents** in this folder (or subfolders)
2. **Run the application** and click "Load data to Pinecone"
3. **The system will automatically**:
   - Load all supported files
   - Process and chunk the content
   - Create vector embeddings
   - Store in Pinecone for searching

## 📝 **File Organization**

You can organize files in subfolders:
```
documents/
├── company-policies/
│   ├── hr-policy.pdf
│   └── security-guidelines.docx
├── product-docs/
│   ├── user-manual.md
│   └── api-reference.txt
└── training-materials/
    └── onboarding-guide.pdf
```

## ⚠️ **Important Notes**

- **File size**: Large files will be automatically chunked
- **Metadata**: Source file path is preserved for citations
- **Updates**: Re-run "Load data" after adding new documents
- **Formats**: Ensure files are in supported formats

## 🔍 **Example Queries**

Once loaded, you can ask questions like:
- "What are the company policies?"
- "How do I use the API?"
- "What's in the training materials?"
