# Configuration management for the Agentic RAG Chatbot

import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import  Field
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables from .env file")
load_dotenv()
logger.info("Environment variables loaded successfully")


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral:7b", env="OLLAMA_MODEL")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    
    # Embedding Model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Reranker Model
    reranker_model: str = Field(default="BAAI/bge-reranker-base", env="RERANKER_MODEL")
    
    # Audio Configuration
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    audio_chunk_size: int = Field(default=1024, env="AUDIO_CHUNK_SIZE")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Retrieval Configuration
    max_chunks: int = Field(default=5, env="MAX_CHUNKS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Agent Configuration
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Performance Configuration
    agent_mode: str = Field(default="fast", env="AGENT_MODE")  # Options: "original" (LangGraph), "fast" (optimized), "ultra_fast" (maximum speed)
    enable_reranking: bool = Field(default=False, env="ENABLE_RERANKING")  # Disable for speed
    max_context_chunks: int = Field(default=3, env="MAX_CONTEXT_CHUNKS")  # Limit context for speed
    response_max_length: int = Field(default=512, env="RESPONSE_MAX_LENGTH")  # Limit response length
    
    # Memory Configuration
    memory_db_path: str = Field(default="memory.db", env="MEMORY_DB_PATH")
    max_memory_items: int = Field(default=1000, env="MAX_MEMORY_ITEMS")
    memory_decay_days: int = Field(default=30, env="MEMORY_DECAY_DAYS")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


# Global settings instance
logger.info("Initializing global settings instance")
settings = Settings()
logger.info("Settings initialized successfully")


def get_settings() -> Settings:
    """Get the global settings instance."""
    logger.debug("Retrieving global settings instance")
    return settings
