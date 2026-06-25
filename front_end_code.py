import os

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Streamlit page configuration
st.set_page_config(
    page_title="The Correlation - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .loading-spinner {
        text-align: center;
        color: #1f77b4;
        font-size: 1.2rem;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🚀 Welcome to The Correlation</div>', unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **The Correlation AI Assistant** 
    
    This intelligent assistant uses:
    - **FAISS Vector Database** for similarity search
    - **HuggingFace Embeddings** for text understanding
    - **Advanced LLM** for generating answers
    
    💡 **How to use:**
    1. Enter your question in the text box
    2. Click the 'Get Answer' button
    3. Wait for the AI to process and respond
    """)
    
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.write("**Embedding Model:** intfloat/e5-base-v2")
    st.write("**LLM Model:** utter-project/eurollm-9b-instruct")
    st.write("**Vector DB:** FAISS")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []


def build_answer(question, context):
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the following context to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    if not api_key:
        return (
            "The configured LLM service is not available, so I’m returning the most relevant retrieved context instead.\n\n"
            + context[:3000]
        )

    try:
        llm = ChatOpenAI(
            temperature=0.3,
            model="utter-project/eurollm-9b-instruct",
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
        )
        prompt = prompt_template.format(context=context[:4000], question=question)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return (
            f"I couldn’t reach the LLM service ({exc}). Here is the available retrieved context instead:\n\n"
            + context[:3000]
        )


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Question input section with enhanced UI
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.subheader("💭 Ask Your Question")
    
    # Text input with placeholder
    user_question = st.text_area(
        "Enter your question here:",
        height=120,
        placeholder="Type your question here... For example: 'What is machine learning?' or 'Explain neural networks...'",
        key="question_input"
    )
    
    # Submit button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submit_button = st.button("🎯 Get Answer", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Statistics and info card
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 15px; 
                color: white;
                text-align: center;'>
        <h3>📈 Quick Stats</h3>
        <p><strong>Documents in DB:</strong> 1000+</p>
        <p><strong>Response Time:</strong> ~5-10s</p>
        <p><strong>Accuracy:</strong> High</p>
        <p><strong>Context Length:</strong> 5 documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features card
    st.markdown("""
    <div style='background-color: #f0f2f6; 
                padding: 1.5rem; 
                border-radius: 15px; 
                margin-top: 1rem;
                border-left: 4px solid #28a745;'>
        <h4>✨ Features</h4>
        <ul style='margin-left: -20px;'>
            <li>Semantic Search</li>
            <li>Context-Aware Answers</li>
            <li>Multiple Document Analysis</li>
            <li>Real-time Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Process the question when submit button is clicked
if submit_button and user_question:
    try:
        # Show loading spinner
        with st.spinner('🔍 Searching through knowledge base...'):
            # Initialize embeddings and database
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/e5-base-v2",
                model_kwargs={"trust_remote_code": True}
            )
            
            db = FAISS.load_local("My_db", embeddings, allow_dangerous_deserialization=True)
            
            # Search for similar documents
            docs = db.similarity_search(user_question, k=5)
            
            # Combine all document content
            all_content = ""
            for doc in docs:
                all_content += doc.page_content + " "
        
        with st.spinner('🤖 Generating intelligent response...'):
            answer_text = build_answer(user_question, all_content)
        
        # Display the answer in a beautiful box
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.subheader("🧠 AI Answer")
        
        # User question display
        st.markdown(f"**Your Question:** {user_question}")
        st.markdown("---")
        
        # AI response
        st.markdown(f"**Answer:** {answer_text}")
        
        # Additional info
        with st.expander("📋 Search Details"):
            st.write(f"**Documents analyzed:** {len(docs)}")
            st.write(f"**Context length:** {len(all_content)} characters")
            st.write("**Sources used:** Similarity search from FAISS database")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add to chat history
        st.session_state.messages.append({
            "question": user_question,
            "answer": answer_text
        })
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Please check if the database files are available and try again.")

elif submit_button and not user_question:
    st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")

# Display chat history if exists
if st.session_state.messages:
    st.markdown("---")
    st.subheader("📜 Conversation History")
    
    for i, msg in enumerate(reversed(st.session_state.messages[-5:]), 1):  # Show last 5 messages
        with st.expander(f"Question {i}: {msg['question'][:50]}..."):
            st.write(f"**Q:** {msg['question']}")
            st.write(f"**A:** {msg['answer']}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Streamlit, LangChain, and FAISS | "
    "The Correlation AI Assistant"
    "</div>",
    unsafe_allow_html=True
)