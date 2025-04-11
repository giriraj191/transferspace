import streamlit as st
from streamlit_float import float_init
from db.retriever import query_faiss_indexes
from db.vectorstore import save_faiss_indexes
from src.pipelines.runnable import get_pipeline
from src.pipelines.doctransformer import transformation_pipeline
import os
import shutil

# Streamlit Page Config
st.set_page_config(page_title="✨ OnlyLinks", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        .main {background-color: #F4F4F4;}
        .stButton>button {border-radius: 10px; background-color: #00AFF0; color: white;}
        .stTextInput>div>div>input, .stNumberInput>div>div>input {border-radius: 8px;}
        
        /* Chat container styling */
        .stChatMessageContent {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 5px;
        }
        
        /* Red button for clear memory */
        .red-btn {background-color: #FF4B4B !important; border-color: #FF4B4B !important;}

        /* Blue button hover effect */
        .stButton>button:hover {
            background-color: white !important; 
            color: #00AFF0 !important;
            border: 2px solid #00AFF0 !important;
        }
        
        /* Fix chat container to keep header visible */
        .chat-container {
            height: 400px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        
        /* Fix input and button alignment */
        .input-container {
            display: flex;
            align-items: center;
        }
        .input-container .stTextInput {
            flex-grow: 1;
            margin-right: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "saved_links" not in st.session_state:
    st.session_state.saved_links = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Function to add feedback
def add_feedback(message, type="info"):
    st.session_state.feedback.append({"message": message, "type": type})

# SIDEBAR LAYOUT
# 1. Enter Links Section
st.sidebar.title("🔗 Website Scraper")
link_input = st.sidebar.text_input("Enter Link")

float_init()

# Equal width buttons for links
col1, col2 = st.sidebar.columns(2)
with col1:
    add_link_btn = st.button("Add Link", use_container_width=True)
with col2:
    process_btn = st.button("Process Links", use_container_width=True)

# 2. Advanced Parameters Section
st.sidebar.subheader("⚙️ Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    depth = st.number_input("Scraping Depth", min_value=1, max_value=5, value=1, step=1)
    num_urls = st.number_input("On Surface Links", min_value=1, max_value=50, value=1, step=1)
with col2:
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=50)

# 3. Clear Buttons side-by-side
clear_cols = st.sidebar.columns(2)
with clear_cols[0]:
    clear_chat = st.button("Clear History", use_container_width=True)
with clear_cols[1]:
    # Red button for Clear Memory
    clear_mem = st.button("Clear Memory", use_container_width=True, 
                          type="primary", help="Clears all saved links and FAISS indexes")
    st.markdown(
        """<style>
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #FF4B4B;
            border-color: #FF4B4B;
        }
        </style>
        """, unsafe_allow_html=True
    )

# 4. Saved Links Section
st.sidebar.subheader("📌 Saved Links")
if st.session_state.saved_links:
    for link in st.session_state.saved_links:
        st.sidebar.write(f"*{link}*")
else:
    st.sidebar.write("No links added yet.")

# 5. Feedback Section (now in sidebar)
st.sidebar.subheader("📢 Notification")
if len(st.session_state.feedback) > 0:
    for feedback in st.session_state.feedback:
        if feedback["type"] == "success":
            st.sidebar.success(feedback["message"])
        elif feedback["type"] == "error":
            st.sidebar.error(feedback["message"])
        elif feedback["type"] == "warning":
            st.sidebar.warning(feedback["message"])
        else:
            st.sidebar.info(feedback["message"])
else:
    st.sidebar.info("No activity yet.")

# Handle adding links
if add_link_btn and link_input:
    st.session_state.saved_links.append(link_input)
    add_feedback(f"Link '{link_input}' added successfully!", "success")
    # Clear link input after adding
    st.session_state.link_input_value = ""
    st.rerun()

# Handle clear chat history
if clear_chat:
    st.session_state.messages = []
    if st.session_state.pipeline and hasattr(st.session_state.pipeline, 'clear_history'):
        st.session_state.pipeline.clear_history()
    st.session_state.pipeline = None
    add_feedback("Chat history cleared!", "info")
    st.rerun()

# Handle clear memory
if clear_mem:
    faiss_dir = os.path.join("db", "faiss_indexes")
    if os.path.exists(faiss_dir):
        for file in os.listdir(faiss_dir):
            file_path = os.path.join(faiss_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    st.session_state.saved_links = []
    st.session_state.feedback = []
    st.session_state.messages = []
    st.session_state.pipeline = None

    add_feedback("Memory cleared successfully!", "success")
    st.rerun()

# Processing Documents
if process_btn and st.session_state.saved_links:
    with st.spinner("*processing documents...*"):
        documents_dict, documents = transformation_pipeline(
            st.session_state.saved_links, depth, num_urls, chunk_size, chunk_overlap
        )
        save_faiss_indexes(documents_dict)
        add_feedback("Processing complete! Links have been indexed and are ready for querying.", "success")
        st.rerun()

# MAIN AREA - CHATBOT UI
st.title("✨ OnlyLinks")
st.markdown("#### Welcome to a Streamlined Space for discussing URLs in Real-Time.")
st.markdown("⬅️ Drop your links in **Website Scraper** to engage in meaningful conversations.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input (new way)
user_input = st.chat_input("Ask a question...")

# Handle User Query
if user_input:
    with st.spinner("*aggregating context, generating response...*"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Get documents and generate response
        retrieved_docs = query_faiss_indexes(user_input)
        if st.session_state.pipeline is None:
            st.session_state.pipeline = get_pipeline(retrieved_docs)        
        response = st.session_state.pipeline(user_input, retrieved_docs)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update UI
        st.rerun()