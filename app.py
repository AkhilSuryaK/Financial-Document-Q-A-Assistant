import streamlit as st
import os
import pandas as pd
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Financial Document Q&A Assistant")
st.markdown("Upload a financial document (PDF or Excel) and ask questions about its content.")

# --- 2. MODEL AND EMBEDDINGS INITIALIZATION ---
# Initialize the Ollama LLM with the specified model
llm = Ollama(model="mistral")
# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="mistral")


# --- 3. DOCUMENT PROCESSING FUNCTIONS ---

def process_document(uploaded_file):
    """
    Processes the uploaded document (PDF or Excel) and returns its text content.
    """
    try:
        # Create a temporary file to store the uploaded content
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        text = ""
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            text = "\n".join(page.page_content for page in pages)
        
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(temp_file_path, engine='openpyxl')
            # Convert the entire DataFrame to a string
            text = df.to_string()
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return text
    
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None


def create_vector_store(text):
    """
    Creates a FAISS vector store from the given text.
    """
    if not text:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.warning("Could not create text chunks. The document might be empty or too small.")
            return None
        
        st.info(f"Created {len(chunks)} text chunks for processing.")
        
        # Create vector store using FAISS
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return vector_store

    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- 4. SESSION STATE MANAGEMENT ---

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- 5. UI COMPONENTS ---

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload your financial document (PDF or Excel)", 
        type=["pdf", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # Process the document only if a new file is uploaded
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            with st.spinner("Processing document... Please wait."):
                # 1. Extract text from the document
                document_text = process_document(uploaded_file)
                
                # 2. Create the vector store
                st.session_state.vector_store = create_vector_store(document_text)

                if st.session_state.vector_store:
                    # 3. Create the retrieval chain
                    prompt = ChatPromptTemplate.from_template("""
                    You are an expert financial analyst. Answer the user's questions accurately based on the provided context.
                    If you don't know the answer, just say that you don't know. Don't make up an answer.
                    <context>
                    {context}
                    </context>
                    Question: {input}
                    """)
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vector_store.as_retriever()
                    st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    st.success(f"Successfully processed '{uploaded_file.name}'!")
                    # Clear previous chat history on new file upload
                    st.session_state.chat_history = []
                else:
                    st.error("Failed to process the document and create a vector store.")
    else:
        # Reset state if no file is uploaded
        st.session_state.vector_store = None
        st.session_state.uploaded_file_name = None
        st.session_state.retrieval_chain = None

# Main chat interface
st.header("Ask a Question")

# Display a message if no document is uploaded
if st.session_state.vector_store is None:
    st.info("Please upload a financial document in the sidebar to begin.")
else:
    # Display previous messages
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                st.markdown(message)
    
    # Get user input
    user_query = st.chat_input("What is the total revenue?")

    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append(user_query)
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get the assistant's response
        with st.spinner("Thinking..."):
            if st.session_state.retrieval_chain:
                response = st.session_state.retrieval_chain.invoke({"input": user_query})
                answer = response.get("answer", "Sorry, I couldn't find an answer.")
                
                # Add assistant response to chat history
                st.session_state.chat_history.append(answer)
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                st.error("Retrieval chain not initialized. Please re-upload the document.")