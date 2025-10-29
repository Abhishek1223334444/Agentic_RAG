# Streamlit UI for the Agentic RAG Chatbot

import streamlit as st
import requests
import requests.exceptions
import json
import io
import base64
import time
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


def post_api_data(endpoint: str, data: Dict[str, Any], timeout: int = 120) -> Optional[Dict[str, Any]]:
    """Post data to API endpoint."""
    try:
        # Longer timeout for chat queries (especially LangGraph mode)
        if "/chat/query" in endpoint:
            timeout = 120  # 2 minutes for chat queries
        elif "/documents/upload" in endpoint:
            timeout = 180  # 3 minutes for document uploads
        
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timed out after {timeout}s. The query may be taking longer than expected.")
        st.info("💡 **Tips to improve response time:**")
        st.info("  • Try using `fast` or `ultra_fast` agent mode in `.env`")
        st.info("  • Simplify your query")
        st.info("  • Ensure Ollama server is running and responsive")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please ensure the FastAPI backend is running.")
        st.info("Run: `python main.py` in your terminal")
        return None
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            st.error(f"⏱️ Request timed out. The server may be slow or overloaded.")
        else:
            st.error(f"Error posting data: {error_msg}")
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
            metadata = message["metadata"]
            with st.expander("📊 Message Details"):
                # Display key metrics
                if "query_time" in metadata:
                    st.metric("⏱️ Response Time", f"{metadata['query_time']:.2f}s")
                
                if "agent_mode" in metadata:
                    mode_badge = "🔷" if metadata["agent_mode"] == "LangGraph" else "⚡"
                    st.write(f"{mode_badge} **Agent Mode**: {metadata['agent_mode']}")
                
                if "iterations" in metadata and metadata["iterations"] is not None:
                    st.write(f"🔄 **Iterations**: {metadata['iterations']}")
                    st.write(f"🔧 **Tools Used**: {metadata.get('tools_used', 0)}")
                
                if "chunks_retrieved" in metadata:
                    st.write(f"📄 **Chunks Retrieved**: {metadata['chunks_retrieved']}")
                
                if "sources_count" in metadata:
                    st.write(f"📚 **Sources**: {metadata['sources_count']}")
                
                # Show full metadata as JSON
                if "full_metadata" in metadata:
                    st.write("**Full Metadata:**")
                    st.json(metadata["full_metadata"])
                else:
                    st.json(metadata)


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
        
        # Agent Mode Information
        st.markdown("## ⚙️ Agent Configuration")
        st.info(
            """
            **Current Mode**: Determined by `AGENT_MODE` in `.env`
            
            **Available Modes**:
            - `original`: LangGraph with multi-step reasoning
            - `fast`: Optimized for speed
            - `ultra_fast`: Maximum speed
            
            Edit `.env` and restart backend to change mode.
            """
        )
        
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
                    status_placeholder = st.empty()
                    with status_placeholder.container():
                        st.info("🤔 Processing query... This may take a moment, especially in LangGraph mode.")
                        
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show progress indicator
                    try:
                        query_data = {
                            "query": user_input,
                            "session_id": st.session_state.session_id,
                            "max_chunks": 5
                        }
                        
                        status_text.text("📡 Connecting to API...")
                        progress_bar.progress(10)
                        
                        status_text.text("🔍 Analyzing query and retrieving context...")
                        progress_bar.progress(30)
                        
                        # Use longer timeout for chat queries
                        response = post_api_data("/chat/query", query_data, timeout=120)
                        
                        if response:
                            status_text.text("✅ Generating response...")
                            progress_bar.progress(90)
                            
                            # Small delay to show completion
                            time.sleep(0.2)
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Complete!")
                            time.sleep(0.5)
                            
                            # Update session ID
                            st.session_state.session_id = response["session_id"]
                            
                            # Extract metadata for display
                            metadata = response.get("metadata", {})
                            
                            # Add assistant response to history
                            assistant_message = {
                                "id": f"assistant_{datetime.now().timestamp()}",
                                "content": response["answer"],
                                "role": "assistant",
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "query_time": response.get("query_time", 0),
                                    "sources_count": len(response.get("sources", [])),
                                    "iterations": metadata.get("iterations", None),
                                    "tools_used": metadata.get("tools_used", 0),
                                    "chunks_retrieved": metadata.get("chunks_retrieved", 0),
                                    "agent_mode": "LangGraph" if metadata.get("iterations") is not None else "Fast/Ultra-Fast",
                                    "full_metadata": metadata
                                }
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            # Rerun to refresh the page (input will be cleared automatically)
                            st.rerun()
                        else:
                            status_text.text("❌ Error occurred")
                            
                    except requests.exceptions.Timeout:
                        status_text.text("⏱️ Request timed out")
                        st.error("⏱️ The request took too long. Try simplifying your query or switching to fast mode.")
                    except Exception as e:
                        status_text.text(f"❌ Error: {str(e)}")
                        st.error(f"An error occurred: {str(e)}")
                    finally:
                        # Clear status indicators
                        progress_bar.empty()
                        status_text.empty()
                        status_placeholder.empty()
        
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
            iterations_list = []
            tools_used_list = []
            agent_modes = []
            
            for message in st.session_state.chat_history:
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    if "query_time" in metadata:
                        query_times.append(metadata["query_time"])
                    if "sources_count" in metadata:
                        source_counts.append(metadata["sources_count"])
                    if "iterations" in metadata and metadata["iterations"] is not None:
                        iterations_list.append(metadata["iterations"])
                    if "tools_used" in metadata:
                        tools_used_list.append(metadata["tools_used"])
                    if "agent_mode" in metadata:
                        agent_modes.append(metadata["agent_mode"])
            
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
                st.metric("⏱️ Avg Response Time", f"{avg_time:.2f}s")
            
            if source_counts:
                avg_sources = sum(source_counts) / len(source_counts)
                st.metric("📚 Avg Sources Used", f"{avg_sources:.1f}")
            
            # LangGraph-specific metrics
            if iterations_list:
                st.markdown("### 🔷 LangGraph Metrics")
                avg_iterations = sum(iterations_list) / len(iterations_list)
                st.metric("🔄 Avg Iterations", f"{avg_iterations:.1f}")
                
                # Iterations chart
                fig_iterations = px.bar(
                    x=list(range(1, len(iterations_list) + 1)),
                    y=iterations_list,
                    title="LangGraph Iterations per Query",
                    labels={"x": "Query Number", "y": "Iterations"}
                )
                st.plotly_chart(fig_iterations, use_container_width=True)
            
            if tools_used_list:
                avg_tools = sum(tools_used_list) / len(tools_used_list)
                st.metric("🔧 Avg Tools Used", f"{avg_tools:.1f}")
            
            # Agent mode distribution
            if agent_modes:
                mode_counts = pd.Series(agent_modes).value_counts()
                if len(mode_counts) > 0:
                    st.markdown("### 📊 Agent Mode Usage")
                    fig_mode = px.pie(
                        values=mode_counts.values,
                        names=mode_counts.index,
                        title="Agent Mode Distribution"
                    )
                    st.plotly_chart(fig_mode, use_container_width=True)
        
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
        st.markdown("### 🔧 Available Agent Tools")
        tools = get_api_data("/tools")
        if tools:
            st.info("These tools are orchestrated by LangGraph in `original` mode")
            for tool in tools["tools"]:
                with st.expander(f"🔧 {tool['name']}"):
                    st.write(f"**Description:** {tool['description']}")
                    if "parameters" in tool:
                        st.write("**Parameters:**")
                        st.json(tool["parameters"])
        else:
            st.info("No tools available. Agent may be running in fast/ultra_fast mode.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Agentic RAG Chatbot - Powered by FastAPI, LangChain + LangGraph, ChromaDB, and Ollama</p>
            <p style='font-size: 0.9em; margin-top: 0.5rem;'>
                🔷 LangGraph Mode | ⚡ Fast Mode | 🚀 Ultra-Fast Mode
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
