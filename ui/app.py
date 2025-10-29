# Streamlit UI for the Agentic RAG Chatbot

import streamlit as st
import requests
import json
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_data(endpoint: str) -> Optional[Dict[str, Any]]:
    """Get data from API endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def post_api_data(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Post data to API endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error posting data: {e}")
        return None


def upload_file_to_api(file) -> Optional[Dict[str, Any]]:
    """Upload file to API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None


def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message."""
    css_class = "user-message" if is_user else "assistant-message"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{'You' if is_user else 'Assistant'}</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show metadata if available
        if "metadata" in message and message["metadata"]:
            with st.expander("Message Details"):
                st.json(message["metadata"])


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the FastAPI backend first.")
        st.info("Run: `python main.py` in your terminal")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # Get system stats
        stats = get_api_data("/stats")
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("documents", 0))
            with col2:
                st.metric("Sessions", stats.get("sessions", 0))
            
            collection_stats = stats.get("collection_stats", {})
            if collection_stats:
                st.metric("Total Chunks", collection_stats.get("total_chunks", 0))
        
        st.markdown("---")
        
        # Document Management
        st.markdown("## 📄 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'txt', 'docx'],
            help="Upload a PDF, TXT, or DOCX file to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    result = upload_file_to_api(uploaded_file)
                    if result:
                        st.success(f"✅ Document uploaded successfully!")
                        st.info(f"Document ID: {result['document_id']}")
                        st.info(f"Processing time: {result['processing_time']:.2f}s")
                        st.session_state.uploaded_files.append(result)
        
        # List uploaded documents
        documents = get_api_data("/documents")
        if documents:
            st.markdown("### Uploaded Documents")
            for doc in documents:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Type:** {doc['document_type']}")
                    st.write(f"**Chunks:** {doc['chunks_count']}")
                    st.write(f"**Uploaded:** {doc['upload_time']}")
                    st.write(f"**Size:** {doc['content_length']} characters")
        
        st.markdown("---")
        
        # Session Management
        st.markdown("## 💬 Session Management")
        
        if st.button("🆕 New Session"):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🗑️ Clear History"):
            if st.session_state.session_id:
                # Delete session from API
                requests.delete(f"{API_BASE_URL}/chat/sessions/{st.session_state.session_id}")
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 Chat Interface")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(message, is_user=(message["role"] == "user"))
        
        # Chat input
        user_input = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main points in the uploaded document?",
            key="user_input"
        )
        
        col_input1, col_input2, col_input3 = st.columns([1, 1, 1])
        
        with col_input1:
            if st.button("🚀 Send", type="primary"):
                if user_input and user_input.strip():
                    # Add user message to history
                    user_message = {
                        "id": f"user_{datetime.now().timestamp()}",
                        "content": user_input,
                        "role": "user",
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(user_message)
                    
                    # Process query
                    with st.spinner("🤔 Thinking..."):
                        query_data = {
                            "query": user_input,
                            "session_id": st.session_state.session_id,
                            "max_chunks": 5
                        }
                        
                        response = post_api_data("/chat/query", query_data)
                        
                        if response:
                            # Update session ID
                            st.session_state.session_id = response["session_id"]
                            
                            # Add assistant response to history
                            assistant_message = {
                                "id": f"assistant_{datetime.now().timestamp()}",
                                "content": response["answer"],
                                "role": "assistant",
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "query_time": response["query_time"],
                                    "sources_count": len(response["sources"]),
                                    "agent_metadata": response["metadata"]
                                }
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            # Clear input
                            st.session_state.user_input = ""
                            st.rerun()
        
        with col_input2:
            if st.button("🎤 Voice Input"):
                st.info("Voice input feature coming soon!")
        
        with col_input3:
            if st.button("🔊 Voice Output"):
                if st.session_state.chat_history:
                    last_message = st.session_state.chat_history[-1]
                    if last_message["role"] == "assistant":
                        with st.spinner("🔊 Generating audio..."):
                            # Synthesize audio
                            audio_response = post_api_data("/audio/synthesize", {
                                "text": last_message["content"],
                                "language": "en"
                            })
                            
                            if audio_response and audio_response.get("audio_data"):
                                # Decode and play audio
                                audio_data = base64.b64decode(audio_response["audio_data"])
                                st.audio(audio_data, format="audio/wav")
    
    with col2:
        st.markdown("## 📈 Analytics & Visualization")
        
        # Query performance metrics
        if st.session_state.chat_history:
            st.markdown("### Query Performance")
            
            # Extract metrics
            query_times = []
            source_counts = []
            
            for message in st.session_state.chat_history:
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    if "query_time" in metadata:
                        query_times.append(metadata["query_time"])
                    if "sources_count" in metadata:
                        source_counts.append(metadata["sources_count"])
            
            if query_times:
                # Query time chart
                fig_time = px.line(
                    x=list(range(1, len(query_times) + 1)),
                    y=query_times,
                    title="Query Response Time",
                    labels={"x": "Query Number", "y": "Time (seconds)"}
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Average metrics
                avg_time = sum(query_times) / len(query_times)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            if source_counts:
                avg_sources = sum(source_counts) / len(source_counts)
                st.metric("Avg Sources Used", f"{avg_sources:.1f}")
        
        # Document analysis
        if documents:
            st.markdown("### Document Analysis")
            
            # Document types pie chart
            doc_types = [doc["document_type"] for doc in documents]
            type_counts = pd.Series(doc_types).value_counts()
            
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Document Types"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Chunks distribution
            chunk_counts = [doc["chunks_count"] for doc in documents]
            if chunk_counts:
                fig_chunks = px.bar(
                    x=[doc["filename"] for doc in documents],
                    y=chunk_counts,
                    title="Chunks per Document",
                    labels={"x": "Document", "y": "Number of Chunks"}
                )
                fig_chunks.update_xaxis(tickangle=45)
                st.plotly_chart(fig_chunks, use_container_width=True)
        
        # System tools
        st.markdown("### Available Tools")
        tools = get_api_data("/tools")
        if tools:
            for tool in tools["tools"]:
                with st.expander(f"🔧 {tool['name']}"):
                    st.write(f"**Description:** {tool['description']}")
                    if "parameters" in tool:
                        st.write("**Parameters:**")
                        st.json(tool["parameters"])
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Agentic RAG Chatbot - Powered by FastAPI, LangChain, ChromaDB, and Ollama</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
