# FastAPI backend for the Agentic RAG Chatbot

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from models import (
    Document, DocumentChunk, ChatMessage, ChatSession, 
    QueryRequest, QueryResponse, DocumentUploadResponse,
    AudioRequest, AudioResponse, DocumentType
)
from embed_store import EmbeddingStore
from llm_agent import LLMAgent
from llm_agent_fast import FastLLMAgent, UltraFastLLMAgent
from llm_agent_context_focused import ContextFocusedAgent
from utils import DocumentProcessor, AudioUtils, generate_id, validate_file_type
from config import settings, get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG Chatbot API",
    description="A production-ready RAG chatbot with multi-step reasoning capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
embedding_store: Optional[EmbeddingStore] = None
llm_agent: Optional[LLMAgent] = None
chat_sessions: Dict[str, ChatSession] = {}
uploaded_documents: Dict[str, Document] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedding_store, llm_agent
    
    try:
        logger.info("Starting Agentic RAG Chatbot API...")
        
        # Initialize embedding store
        embedding_store = EmbeddingStore()
        
        # Initialize LLM agent based on configuration
        agent_mode = settings.agent_mode.lower()
        logger.info(f"Initializing LLM agent in '{agent_mode}' mode")
        
        if agent_mode == "context_focused":
            llm_agent = ContextFocusedAgent(embedding_store)
            logger.info("Context-Focused LLM Agent initialized")
        elif agent_mode == "ultra_fast":
            llm_agent = UltraFastLLMAgent(embedding_store)
            logger.info("Ultra-Fast LLM Agent initialized")
        elif agent_mode == "fast":
            llm_agent = FastLLMAgent(embedding_store)
            logger.info("Fast LLM Agent initialized")
        else:
            llm_agent = LLMAgent(embedding_store)
            logger.info("Original LLM Agent initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Agentic RAG Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    try:
        # Check if components are initialized
        if embedding_store is None or llm_agent is None:
            logger.warning("Service components not initialized")
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Get collection stats
        stats = embedding_store.get_collection_stats()
        logger.info(f"Health check passed - Collection stats: {stats}")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "embedding_store": "ready",
                "llm_agent": "ready"
            },
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document."""
    logger.info(f"Document upload request received: {file.filename} ({file.size} bytes)")
    try:
        # Validate file type
        allowed_extensions = [".pdf", ".txt", ".docx"]
        if not validate_file_type(file.filename, allowed_extensions):
            logger.warning(f"Unsupported file type attempted: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )
        
        # Check file size (limit to 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            logger.warning(f"File too large: {file.filename} ({file.size} bytes)")
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 10MB."
            )
        
        # Read file content
        content = await file.read()
        logger.info(f"File content read successfully: {len(content)} bytes")
        
        # Determine file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == ".pdf":
            doc_type = DocumentType.PDF
        elif file_extension == ".txt":
            doc_type = DocumentType.TXT
        elif file_extension == ".docx":
            doc_type = DocumentType.DOCX
        else:
            logger.warning(f"Unsupported file extension: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary file created: {temp_file_path}")
        
        try:
            # Extract text from document
            start_time = datetime.now()
            logger.info(f"Extracting text from {doc_type.value} file")
            text_content = DocumentProcessor.extract_text(temp_file_path, doc_type.value)
            
            # Create document object
            document = Document(
                id=generate_id(),
                filename=file.filename,
                content=text_content,
                document_type=doc_type,
                upload_time=datetime.now()
            )
            
            logger.info(f"Document created with ID: {document.id}, content length: {len(text_content)} characters")
            
            # Process document in background
            background_tasks.add_task(process_document_background, document)
            logger.info(f"Background processing task added for document: {document.id}")
            
            # Store document reference
            uploaded_documents[document.id] = document
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document upload completed in {processing_time:.2f} seconds")
            
            return DocumentUploadResponse(
                document_id=document.id,
                filename=document.filename,
                chunks_created=0,  # Will be updated after background processing
                processing_time=processing_time,
                status="uploaded"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            logger.info(f"Temporary file cleaned up: {temp_file_path}")
            
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_document_background(document: Document):
    """Process document in background."""
    logger.info(f"Starting background processing for document {document.id}: {document.filename}")
    try:
        start_time = datetime.now()
        
        # Add document to embedding store
        logger.info(f"Adding document {document.id} to embedding store")
        embedding_store.add_document(document)
        
        # Get chunks count
        chunks = embedding_store.get_document_chunks(document.id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Document {document.id} processed successfully with {len(chunks)} chunks in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing document {document.id}: {e}")
        # Update document status to indicate processing failure
        if document.id in uploaded_documents:
            uploaded_documents[document.id].status = "processing_failed"


@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """List all uploaded documents."""
    logger.info("Document list request received")
    try:
        documents = []
        for doc_id, document in uploaded_documents.items():
            # Get chunk count
            chunks = embedding_store.get_document_chunks(doc_id)
            
            documents.append({
                "id": document.id,
                "filename": document.filename,
                "document_type": document.document_type.value,
                "upload_time": document.upload_time.isoformat(),
                "chunks_count": len(chunks),
                "content_length": len(document.content)
            })
        
        logger.info(f"Returning {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    logger.info(f"Document details request for ID: {document_id}")
    try:
        if document_id not in uploaded_documents:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = uploaded_documents[document_id]
        chunks = embedding_store.get_document_chunks(document_id)
        
        logger.info(f"Retrieved document {document_id} with {len(chunks)} chunks")
        
        return {
            "id": document.id,
            "filename": document.filename,
            "document_type": document.document_type.value,
            "upload_time": document.upload_time.isoformat(),
            "chunks": [chunk.dict() for chunk in chunks],
            "content_length": len(document.content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    logger.info(f"Document deletion request for ID: {document_id}")
    try:
        if document_id not in uploaded_documents:
            logger.warning(f"Document not found for deletion: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from embedding store
        logger.info(f"Deleting document {document_id} from embedding store")
        success = embedding_store.delete_document(document_id)
        
        # Remove from uploaded documents
        del uploaded_documents[document_id]
        
        logger.info(f"Document {document_id} deleted successfully")
        return {"success": success, "message": f"Document {document_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/chat/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """Process a chat query."""
    logger.info(f"Chat query received: session_id={request.session_id}, query_length={len(request.query)}")
    try:
        if llm_agent is None:
            logger.error("LLM agent not initialized")
            raise HTTPException(status_code=503, detail="LLM agent not initialized")
        
        # Process query with agent
        start_time = datetime.now()
        logger.info(f"Processing query with LLM agent")
        response = llm_agent.process_query(request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query processed in {processing_time:.2f} seconds, response length: {len(response.answer)}")
        
        # Store in session if session_id provided
        if request.session_id:
            if request.session_id not in chat_sessions:
                logger.info(f"Creating new chat session: {request.session_id}")
                chat_sessions[request.session_id] = ChatSession(
                    id=request.session_id,
                    messages=[]
                )
            
            # Add messages to session
            session = chat_sessions[request.session_id]
            session.messages.append(ChatMessage(
                id=generate_id(),
                content=request.query,
                role="user",
                timestamp=datetime.now()
            ))
            session.messages.append(ChatMessage(
                id=generate_id(),
                content=response.answer,
                role="assistant",
                timestamp=datetime.now()
            ))
            session.updated_at = datetime.now()
            logger.info(f"Messages added to session {request.session_id}, total messages: {len(session.messages)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session history."""
    logger.info(f"Chat session request for ID: {session_id}")
    try:
        if session_id not in chat_sessions:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = chat_sessions[session_id]
        logger.info(f"Retrieved session {session_id} with {len(session.messages)} messages")
        
        return {
            "id": session.id,
            "messages": [msg.dict() for msg in session.messages],
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    logger.info(f"Chat session deletion request for ID: {session_id}")
    try:
        if session_id not in chat_sessions:
            logger.warning(f"Session not found for deletion: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        del chat_sessions[session_id]
        logger.info(f"Session {session_id} deleted successfully")
        return {"success": True, "message": f"Session {session_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.post("/audio/transcribe", response_model=AudioResponse)
async def transcribe_audio(audio_request: AudioRequest):
    """Transcribe audio to text."""
    logger.info(f"Audio transcription request received: {len(audio_request.audio_data)} bytes")
    try:
        if llm_agent is None:
            logger.error("LLM agent not initialized for audio transcription")
            raise HTTPException(status_code=503, detail="LLM agent not initialized")
        
        start_time = datetime.now()
        
        # Get voice tool
        voice_tool = llm_agent.tool_registry.get_tool("voice_tool")
        if not voice_tool:
            logger.error("Voice tool not available")
            raise HTTPException(status_code=503, detail="Voice tool not available")
        
        # Execute speech-to-text
        logger.info("Executing speech-to-text conversion")
        result = voice_tool.execute(
            operation="speech_to_text",
            data=audio_request.audio_data,
            language="en"  # Default to English
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result.get("success"):
            logger.info(f"Audio transcribed successfully in {processing_time:.2f} seconds: {len(result.get('text', ''))} characters")
        else:
            logger.warning(f"Audio transcription failed: {result.get('error')}")
        
        return AudioResponse(
            text=result.get("text", ""),
            audio_data=None,
            processing_time=processing_time,
            success=result.get("success", False),
            error_message=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/audio/synthesize", response_model=AudioResponse)
async def synthesize_audio(text: str, language: str = "en"):
    """Synthesize text to speech."""
    logger.info(f"Audio synthesis request received: {len(text)} characters, language: {language}")
    try:
        if llm_agent is None:
            logger.error("LLM agent not initialized for audio synthesis")
            raise HTTPException(status_code=503, detail="LLM agent not initialized")
        
        start_time = datetime.now()
        
        # Get voice tool
        voice_tool = llm_agent.tool_registry.get_tool("voice_tool")
        if not voice_tool:
            logger.error("Voice tool not available")
            raise HTTPException(status_code=503, detail="Voice tool not available")
        
        # Execute text-to-speech
        logger.info("Executing text-to-speech conversion")
        result = voice_tool.execute(
            operation="text_to_speech",
            data=text,
            language=language
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result.get("success"):
            audio_size = len(result.get("audio_data", b""))
            logger.info(f"Audio synthesized successfully in {processing_time:.2f} seconds: {audio_size} bytes")
        else:
            logger.warning(f"Audio synthesis failed: {result.get('error')}")
        
        return AudioResponse(
            text=text,
            audio_data=result.get("audio_data"),
            processing_time=processing_time,
            success=result.get("success", False),
            error_message=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error synthesizing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/tools")
async def get_available_tools():
    """Get available agent tools."""
    logger.info("Available tools request received")
    try:
        if llm_agent is None:
            logger.error("LLM agent not initialized for tools request")
            raise HTTPException(status_code=503, detail="LLM agent not initialized")
        
        tools = llm_agent.get_available_tools()
        logger.info(f"Returning {len(tools)} available tools")
        return {"tools": tools}
        
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    logger.info("System stats request received")
    try:
        stats = {
            "documents": len(uploaded_documents),
            "sessions": len(chat_sessions),
            "collection_stats": embedding_store.get_collection_stats() if embedding_store else {},
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Returning system stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/reset")
async def reset_system():
    """Reset the system (delete all data)."""
    logger.warning("System reset request received")
    try:
        # Reset embedding store
        if embedding_store:
            logger.info("Resetting embedding store collection")
            embedding_store.reset_collection()
        
        # Clear uploaded documents
        doc_count = len(uploaded_documents)
        uploaded_documents.clear()
        logger.info(f"Cleared {doc_count} uploaded documents")
        
        # Clear chat sessions
        session_count = len(chat_sessions)
        chat_sessions.clear()
        logger.info(f"Cleared {session_count} chat sessions")
        
        logger.warning("System reset completed successfully")
        return {"success": True, "message": "System reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
