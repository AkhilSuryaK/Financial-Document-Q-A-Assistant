# 💡 Financial Document Q&A Assistant - Unlock Insights from Your Financial Documents

An AI-powered web application that allows users to **upload financial documents (PDF/Excel)** and **ask questions in natural language**. Built to assist **analysts, accountants, and finance professionals** in quickly extracting key information from complex financial files.

> 🔗 Live Demo: _Coming Soon_  
> 👨‍💻 Developed by [Kolipaka Akhil Surya](https://www.linkedin.com/in/akhilsurya/)

---

## 🧠 About the Application

This project leverages **LangChain, Ollama (Mistral), and FAISS** to provide a context-aware Q&A system over financial documents. Users can upload **PDF or Excel files**, and the assistant will answer queries based solely on the uploaded content.  

It supports **semantic search over large documents**, splitting content into chunks to ensure accurate retrieval and answering.

---

## ✨ Features

- 📌 Upload **PDF or Excel financial documents**  
- 💬 Ask questions in **natural language**  
- 🧠 Context-aware answers using **Ollama Mistral LLM**  
- 📚 Processes large documents with **FAISS vector store**  
- 🔄 Maintains **chat history** for ongoing queries  

---

## 🔧 Tech Stack

- **Frontend / UI:** Streamlit  
- **Backend / AI:** Python, LangChain, Ollama (Mistral)  
- **Document Processing:** PyPDF, Pandas, OpenPyXL  
- **Vector Database:** FAISS  
- **Embeddings:** OllamaEmbeddings  

---

## 📁 Project Structure

Financial-Document-QA/

├── app.py # Main Streamlit application  
├── requirements.txt # Project dependencies  
├── README.md # Project documentation  
└── temp/ # Temporary folder for uploaded files  

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/AkhilSuryaK/Financial-Document-QA-Assistant.git

# Navigate into the project
cd Financial-Document-QA-Assistant

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull mistral

# Run the application
streamlit run app.py

🚀 Usage

Ensure Ollama is running in the background.

Open the application in your browser (automatically opened by Streamlit).

Upload a financial document (PDF or Excel) in the sidebar.

Wait while the document is processed and a vector store is created.

Ask questions like:

“What is the total revenue?”

“What were the expenses in Q2?”

Get context-aware answers instantly.

🤝 Contributing

Contributions are welcome! Follow these steps:

Fork the repository

Create a new feature branch:

git checkout -b feature-name

Make your changes and commit:
git commit -m "Added new feature"

Push and create a Pull Request:
git push origin feature-name

Please follow clean code practices and test before submitting PRs.

📬 Contact

GitHub: @AkhilSuryaK
LinkedIn: Akhil Surya Kolipaka

📃 License

This project is licensed under the MIT License.
