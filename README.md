# 🧠 LumiDoc AI
### Your Intelligent Document Assistant 🚀
**Powered by Luminex Technologies**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## 📖 Overview

**LumiDoc AI** is a powerful Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents. Built with **Streamlit** and **LangChain**, it utilizes the **Google Flan-T5** model to provide accurate, context-aware answers from your uploaded files.

Whether you are a student researching a thesis or a professional analyzing reports, LumiDoc AI turns static PDFs into interactive conversations.

## ✨ Features

* **📄 Multi-PDF Support:** Upload multiple PDF documents at once.
* **🔍 Advanced Text Extraction:** Uses `PyPDF2` for efficient text parsing.
* **🧠 Intelligent Chunking:** Splits large documents into manageable chunks using `LangChain`.
* **⚡ Vector Search:** Utilizes **FAISS** (Facebook AI Similarity Search) for fast and accurate information retrieval.
* **🤖 AI-Powered Chat:** Powered by `google/flan-t5-large` for generating human-like responses.
* **💬 Persistent Chat History:** Keeps track of your conversation context.
* **🎨 Professional UI:** A clean, dark-themed interface designed for specialized use cases.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Flan-T5 (via HuggingFace Hub)
* **Embeddings:** Sentence Transformers (`all-MiniLM-L12-v2`)
* **Vector Store:** FAISS (CPU)
* **Orchestration:** LangChain

---

## 🚀 Installation & Local Setup

Follow these steps to run the application locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/UHDNMJayalath/Luminex-pdf-chat.git](https://github.com/UHDNMJayalath/Luminex-pdf-chat.git)
cd Luminex-pdf-chat
