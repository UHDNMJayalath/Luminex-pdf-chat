import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Page Configuration ---
st.set_page_config(page_title="LumiDoc AI", page_icon="🧠", layout="wide")

# --- Custom CSS for UI and Footer ---
st.markdown("""
<style>
    /* Chat Input Styling */
    .stTextInput > div > div > input {
        background-color: #2b313e;
        color: white;
        border-radius: 10px;
    }
    
    /* Sidebar Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* --- FIXED FOOTER STYLING --- */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #222831;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #393E46;
        z-index: 100;
    }
    
    /* Adjust main content padding so it doesn't get hidden behind the footer */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 60px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Extract text from PDF
def get_pdf_text(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 2. Split text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# 3. Create Vector Store (FAISS)
def get_vectorstore(text_chunks):
    # Load embeddings (this might take a moment)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# 4. Create Conversation Chain (Lazy loading imports to prevent memory crashes)
def get_conversation_chain(vectorstore):
    # Import heavy libraries only when needed
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"Device selected: {device}") 

    # Using 'flan-t5-large'. Change to 'google/flan-t5-base' if RAM is low.
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large", 
        max_new_tokens=512,
        temperature=0.3,
        device=device
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Memory for context retention
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )

# --- Main Application ---
def main():
    load_dotenv()
    
    # --- Footer Display ---
    st.markdown("""
    <div class="footer">
        <p>
            © 2025 <b>Luminex Technologies</b>. All Rights Reserved. 
            | Developed by <a href="#" style="color: #00ADB5; text-decoration: none;">Nishaka Mahesh</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # --- Sidebar Section ---
    with st.sidebar:
        # Logo Display
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=70)
        
        st.markdown("## 📚 LumiDoc Hub")
        st.caption("By **Luminex Technologies**")
        st.markdown("---")
        
        st.subheader("📂 Upload Documents")
        st.markdown("Upload your PDFs here to start chatting.")
        
        # File Uploader
        pdf_docs = st.file_uploader( 
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            label_visibility="collapsed"
        )
        
        st.markdown("") 
        
        # Process Button (Primary Color)
        if st.button("🚀 Process Files", type="primary"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
            else:
                # Progress Status Bar
                with st.status("⚙️ Processing documents...", expanded=True) as status:
                    st.write("Reading PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    
                    st.write("Splitting text into chunks...")
                    text_chunks = get_text_chunks(raw_text)
                    
                    st.write("Creating Vector Store (This may take a while)...")
                    vectorstore = get_vectorstore(text_chunks)
                    
                    st.write("Initializing AI Model...")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    st.session_state.processComplete = True
        
        st.markdown("---")
        # Clear Chat Button
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = None
            st.rerun()

    # --- Main Chat Interface ---
    
    # Header Layout
    col1, col2 = st.columns([1, 15])
    with col1:
         st.markdown("<h1>🧠</h1>", unsafe_allow_html=True)
    with col2:
        st.title("LumiDoc AI")
        st.markdown("#### Your Intelligent Document Assistant | Powered by **Luminex Technologies**")
    
    st.divider()

    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        # Show Welcome Message if chat history is empty
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #888;'>
                <h2>👋 Welcome to LumiDoc AI!</h2>
                <p>I'm here to help you analyze your PDF documents easily.</p>
                <p style='font-size: 0.9rem;'>👈 To get started, please upload your PDFs in the sidebar and click <b>'Process Files'</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.processComplete:
                 st.info("✅ Documents processed! You can now ask questions below. 👇")

        # Display Chat History
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.write(message.content)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message.content)

    # Chat Input Area
    if user_question := st.chat_input("Ask a question about your documents..."):
        if st.session_state.conversation is None:
            st.error("⚠️ Please upload and process documents first!")
        else:
            # Display User Message
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(user_question)
                
            # Display Bot Response with Loading Spinner
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': user_question})
                    st.session_state.chat_history = response['chat_history']
                    st.write(response['chat_history'][-1].content)

if __name__ == "__main__":
    main()