# Agentic RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that can understand, retrieve, reason over, and converse about uploaded documents using FastAPI, LangChain + LangGraph, ChromaDB, and Ollama (Mistral 7B) for fully offline inference.

## 🚀 Features

- **Document Processing**: Upload PDFs, extract and chunk text, generate embeddings
- **Vector Storage**: ChromaDB for efficient similarity search
- **Agentic Reasoning**: Multi-step reasoning with LangGraph (fully functional)
- **Multiple Agent Modes**: Choose between LangGraph, optimized fast mode, or ultra-fast mode
- **Offline LLM**: Ollama with Mistral 7B for local inference
- **Advanced Retrieval**: BGE reranker for improved relevance (optional)
- **Voice Integration**: Whisper for speech-to-text, gTTS for text-to-speech
- **Conversational Memory**: Maintains context across conversations
- **Modern UI**: Streamlit interface with chat, upload, and visualization
- **Modular Architecture**: Extensible design for future enhancements

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed and running
- At least 8GB RAM (for Mistral 7B model)
- 2GB free disk space

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Agentic_RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
```

### 5. Download Mistral 7B Model
```bash
ollama pull mistral:7b
```

### 6. Configure Environment
```bash
cp env_example.txt .env
# Edit .env file with your preferred settings
```

## 🚀 Quick Start

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Start FastAPI Backend
```bash
python main.py
```

### 3. Start Streamlit UI (in another terminal)
```bash
streamlit run ui/app.py
```

### 4. Access the Application
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
├── main.py                 # FastAPI application entry point
├── llm_agent.py           # LangGraph agent implementation (original mode)
├── embed_store.py         # ChromaDB embedding store
├── memory.py              # Conversational memory system
├── config.py              # Configuration management
├── requirements.txt        # Python dependencies
├── env_example.txt         # Environment configuration template
├── tools/                 # Agent tools
│   └── __init__.py        # Tool registry and implementations
├── ui/                    # Streamlit UI
│   └── app.py             # Main UI application
├── models/                # Pydantic models
│   └── __init__.py        # Data models
├── utils/                 # Utility functions
│   └── __init__.py        # Helper functions
└── README.md              # This file
```

## 💡 Usage Examples

### Document Upload and Query
1. Upload a PDF document through the Streamlit interface
2. Wait for processing and embedding generation
3. Ask questions about the document content
4. The agent will retrieve relevant chunks and provide contextual answers

### Voice Interaction
1. Click the microphone button to record your question
2. The system will transcribe your speech using Whisper
3. Process the query and generate a response
4. Listen to the audio response using gTTS

### Multi-step Reasoning (LangGraph Mode)
When using `AGENT_MODE=original`, the agent can handle complex queries that require multiple steps:
- "Compare the main points in chapters 1 and 3"
- "Summarize the key findings and then explain their implications"
- "Find information about X, then search for related concepts Y"

The LangGraph agent automatically:
- Analyzes the query to determine which tools are needed
- Executes tools in the correct sequence
- Combines results from multiple tools
- Generates comprehensive responses

**Note**: Multi-step reasoning is available only in `original` mode. For faster responses with simpler queries, use `fast` or `ultra_fast` modes.

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=documents

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reranker Model
RERANKER_MODEL=BAAI/bge-reranker-base

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501

# Retrieval Configuration
MAX_CHUNKS=3
CHUNK_SIZE=300
CHUNK_OVERLAP=50

# Agent Configuration
MAX_ITERATIONS=10
TEMPERATURE=0.7
AGENT_MODE=original
# Options: "original" (LangGraph with multi-step reasoning), "fast" (optimized), "ultra_fast" (maximum speed)

# Performance Configuration (optional)
ENABLE_RERANKING=false
MAX_CONTEXT_CHUNKS=3
RESPONSE_MAX_LENGTH=512
```

### Agent Modes

The system supports three agent modes, configured via `AGENT_MODE` in `.env`:

| Mode | Description | Best For |
|------|-------------|----------|
| **`original`** | Full LangGraph implementation with multi-step reasoning, tool orchestration, and conditional routing | Complex queries requiring multi-step reasoning, tool coordination |
| **`fast`** | Optimized agent without LangGraph overhead, direct retrieval and response | General queries with good balance of speed and accuracy |
| **`ultra_fast`** | Maximum speed mode with minimal context and short responses | Simple queries requiring fastest response times |

**Changing Agent Mode:**
```bash
# Edit .env file
AGENT_MODE=original    # For LangGraph
AGENT_MODE=fast        # For optimized speed (default)
AGENT_MODE=ultra_fast  # For maximum speed
```

### Model Configuration
- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` by default
- **Reranker**: Uses `BAAI/bge-reranker-base` for improved relevance (can be disabled for speed)
- **LLM**: Mistral 7B via Ollama (fully offline)

## 🔧 API Endpoints

### Document Management
- `POST /documents/upload` - Upload a document
- `GET /documents` - List all documents
- `GET /documents/{document_id}` - Get document details
- `DELETE /documents/{document_id}` - Delete a document

### Chat Interface
- `POST /chat/query` - Process a chat query
- `GET /chat/sessions/{session_id}` - Get chat session history
- `DELETE /chat/sessions/{session_id}` - Delete a chat session

### Audio Processing
- `POST /audio/transcribe` - Convert speech to text
- `POST /audio/synthesize` - Convert text to speech

### System Management
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /tools` - Available agent tools
- `POST /reset` - Reset the system

## 🧠 Agent Tools

The system includes several specialized tools that work seamlessly in LangGraph mode:

1. **Retriever Tool**: Searches for relevant document chunks using vector similarity
2. **Summarizer Tool**: Creates summaries of content using the LLM
3. **Comparator Tool**: Compares different pieces of content
4. **Voice Tool**: Handles speech-to-text and text-to-speech

### LangGraph Agent Architecture

When using `AGENT_MODE=original`, the system leverages LangGraph for sophisticated workflow orchestration:

- **Conditional Routing**: Agent intelligently routes to appropriate tools based on query analysis
- **Multi-step Reasoning**: Handles complex queries requiring multiple tool invocations
- **Tool Execution Tracking**: Prevents infinite loops and re-execution of tools
- **Proper Termination**: Ensures graphs complete within iteration limits
- **Error Handling**: Graceful error recovery throughout the workflow

The LangGraph workflow follows this pattern:
```
Query → Agent Node (Analyze) → Conditional Routing → Tools → Agent Node → Finalize → Response
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Model not found**
   ```bash
   # Pull the model
   ollama pull mistral:7b
   ```

3. **Port conflicts**
   - Change ports in `.env` file
   - Ensure ports 8000 and 8501 are available

4. **Memory issues**
   - Reduce `MAX_CHUNKS` in configuration
   - Use a smaller embedding model
   - Increase system RAM
   - Switch to `fast` or `ultra_fast` agent mode

5. **LangGraph recursion errors**
   - ✅ **Fixed!** The recursion limit issue has been resolved
   - If you encounter recursion errors, check that you're using the latest version
   - The system automatically sets recursion_limit to 50 and caps max_iterations at 10
   - Ensure `AGENT_MODE=original` is set correctly in `.env`

### Performance Optimization

1. **For better performance with LangGraph** (`AGENT_MODE=original`):
   - Use GPU acceleration for embeddings (if available)
   - Increase `CHUNK_SIZE` for longer documents
   - Adjust `MAX_ITERATIONS` based on query complexity (default: 10)
   - Disable reranking: `ENABLE_RERANKING=false` for faster responses

2. **For maximum speed**:
   - Use `AGENT_MODE=fast` or `AGENT_MODE=ultra_fast`
   - Set `ENABLE_RERANKING=false` (saves 2-3 seconds per query)
   - Reduce `MAX_CONTEXT_CHUNKS=3` (default)
   - Lower `RESPONSE_MAX_LENGTH=512` (default)

3. **For lower resource usage**:
   - Use smaller models (e.g., `llama3.2:3b` instead of `mistral:7b`)
   - Reduce `MAX_CHUNKS` to 3-5
   - Switch to `ultra_fast` mode
   - Enable memory cleanup

## 🧪 Testing

```bash
# Run basic health check
curl http://localhost:8000/health

# Test document upload
curl -X POST "http://localhost:8000/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.pdf"

# Test chat query
curl -X POST "http://localhost:8000/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?"}'
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
   ```bash
   # Use production-grade settings
   export OLLAMA_BASE_URL=http://your-ollama-server:11434
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

2. **Process Management**:
   ```bash
   # Use PM2 or similar for process management
   pm2 start main.py --name "rag-chatbot"
   ```

3. **Reverse Proxy**:
   - Use Nginx or Apache for reverse proxy
   - Configure SSL certificates
   - Set up load balancing if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP8 standards
- Use type hints throughout
- Write comprehensive docstrings
- Include unit tests for core functionality

## 🔮 Future Enhancements

- **Multi-agent Collaboration**: Multiple specialized agents working together
- **Web Search Integration**: Real-time web search capabilities
- **Retrieval Evaluation**: Metrics for retrieval quality assessment
- **Advanced Memory**: Semantic memory with embeddings
- **Custom Models**: Support for custom fine-tuned models
- **API Rate Limiting**: Production-ready rate limiting
- **Authentication**: User authentication and authorization
- **Multi-language Support**: Support for multiple languages

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the UI framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
