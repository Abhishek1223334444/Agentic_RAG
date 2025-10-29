# Fast LLM Agent - Optimized for Speed
# Direct retrieval and response without LangGraph overhead

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage

from models import QueryRequest, QueryResponse
from embed_store import EmbeddingStore
from config import settings
from utils import generate_id

logger = logging.getLogger(__name__)


class FastLLMAgent:
    """Fast LLM agent optimized for speed with minimal overhead."""
    
    def __init__(self, embedding_store: EmbeddingStore):
        """Initialize the fast agent."""
        logger.info("Initializing Fast LLM Agent...")
        self.embedding_store = embedding_store
        self.llm = None
        self._initialize_llm()
        logger.info("Fast LLM Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize Ollama LLM with optimized settings."""
        try:
            logger.info(f"Initializing Ollama LLM: {settings.ollama_model}")
            self.llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.3,  # Lower temperature for faster, more focused responses
                num_predict=512,   # Limit response length for speed
                num_ctx=2048,      # Smaller context window for speed
                top_k=20,          # Reduce sampling options for speed
                top_p=0.8,         # Reduce sampling complexity
                repeat_penalty=1.1, # Prevent repetition
                stop=["Human:", "Assistant:", "\n\n"]  # Stop tokens for faster completion
            )
            logger.info("Ollama LLM initialized with optimized settings")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query with minimal overhead for maximum speed."""
        logger.info(f"Fast processing query: {request.query[:50]}...")
        start_time = datetime.now()
        
        try:
            # Step 1: Fast retrieval (no reranking for speed)
            logger.debug("Performing fast retrieval")
            chunks = self._fast_retrieve(request.query)
            
            # Step 2: Generate response with minimal context
            logger.debug("Generating fast response")
            response_text = self._generate_fast_response(request.query, chunks)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = QueryResponse(
                answer=response_text,
                sources=[],  # Skip source conversion for speed
                session_id=request.session_id or generate_id(),
                query_time=processing_time,
                metadata={
                    "mode": "fast",
                    "chunks_retrieved": len(chunks),
                    "context_length": sum(len(chunk.content) for chunk in chunks),
                    "optimization": "speed_optimized"
                }
            )
            
            logger.info(f"Fast query processed in {processing_time:.2f}s: {len(chunks)} chunks, {len(response_text)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Error in fast query processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                answer=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                session_id=request.session_id or generate_id(),
                query_time=processing_time,
                metadata={"error": str(e), "mode": "fast"}
            )
    
    def _fast_retrieve(self, query: str) -> List[Any]:
        """Fast retrieval without reranking for maximum speed."""
        logger.debug(f"Fast retrieval for query: {query[:30]}...")
        
        try:
            # Use similarity search with lower threshold for better recall
            chunks = self.embedding_store.similarity_search(
                query=query,
                threshold=0.3,  # Lower threshold for better recall
                max_results=3    # Fewer chunks for speed
            )
            
            logger.debug(f"Fast retrieval completed: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in fast retrieval: {e}")
            return []
    
    def _generate_fast_response(self, query: str, chunks: List[Any]) -> str:
        """Generate response with optimized prompt and minimal context."""
        logger.debug("Generating fast response with minimal context")
        
        try:
            # Prepare context from all chunks (not just top 2)
            context = ""
            if chunks:
                context_parts = []
                for i, chunk in enumerate(chunks):
                    # Use more content from each chunk
                    context_parts.append(f"Source {i+1}: {chunk.content[:500]}...")
                context = "\n\n".join(context_parts)
                logger.debug(f"Prepared context from {len(chunks)} chunks: {len(context)} characters")
            
            # Much more explicit prompt that forces using the exact context
            if context:
                prompt = f"""IMPORTANT: You must answer using ONLY the exact information provided in the context below. Do not add any external knowledge. If the context contains the answer, copy and use that exact information.

Question: {query}

Context:
{context}

Instructions:
1. Read the context carefully
2. Find the relevant information that answers the question
3. Use the EXACT text from the context
4. Do not add any information not present in the context

Answer:"""
            else:
                prompt = f"""Answer this question briefly and directly.

Question: {query}

Answer:"""
            
            logger.debug(f"Generated optimized prompt: {len(prompt)} characters")
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            logger.debug(f"Generated response: {len(response)} characters")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating fast response: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return minimal tool info for compatibility."""
        return [
            {
                "name": "fast_retriever",
                "description": "Fast document retrieval without reranking",
                "parameters": {"query": "string"}
            }
        ]


class UltraFastLLMAgent:
    """Ultra-fast agent with even more aggressive optimizations."""
    
    def __init__(self, embedding_store: EmbeddingStore):
        """Initialize the ultra-fast agent."""
        logger.info("Initializing Ultra-Fast LLM Agent...")
        self.embedding_store = embedding_store
        self.llm = None
        self._initialize_llm()
        logger.info("Ultra-Fast LLM Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize Ollama LLM with ultra-optimized settings."""
        try:
            logger.info(f"Initializing Ultra-Fast Ollama LLM: {settings.ollama_model}")
            self.llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,   # Very low temperature for deterministic responses
                num_predict=256,   # Very short responses for speed
                num_ctx=1024,      # Small context window
                top_k=10,          # Minimal sampling options
                top_p=0.7,         # Reduced sampling complexity
                repeat_penalty=1.05, # Minimal repetition penalty
                stop=["Human:", "Assistant:", "\n\n", ".", "!", "?"]  # More stop tokens
            )
            logger.info("Ultra-Fast Ollama LLM initialized")
        except Exception as e:
            logger.error(f"Error initializing ultra-fast LLM: {e}")
            raise
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Ultra-fast query processing with maximum optimizations."""
        logger.info(f"Ultra-fast processing query: {request.query[:30]}...")
        start_time = datetime.now()
        
        try:
            # Ultra-fast retrieval (single best chunk only)
            logger.debug("Performing ultra-fast retrieval")
            chunks = self._ultra_fast_retrieve(request.query)
            
            # Ultra-fast response generation
            logger.debug("Generating ultra-fast response")
            response_text = self._generate_ultra_fast_response(request.query, chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = QueryResponse(
                answer=response_text,
                sources=[],
                session_id=request.session_id or generate_id(),
                query_time=processing_time,
                metadata={
                    "mode": "ultra_fast",
                    "chunks_retrieved": len(chunks),
                    "optimization": "maximum_speed"
                }
            )
            
            logger.info(f"Ultra-fast query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in ultra-fast processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                session_id=request.session_id or generate_id(),
                query_time=processing_time,
                metadata={"error": str(e), "mode": "ultra_fast"}
            )
    
    def _ultra_fast_retrieve(self, query: str) -> List[Any]:
        """Ultra-fast retrieval - single best chunk only."""
        logger.debug("Ultra-fast retrieval (single chunk)")
        
        try:
            # Get only the single best chunk with lower threshold
            chunks = self.embedding_store.similarity_search(
                query=query,
                threshold=0.2,     # Very low threshold for better recall
                max_results=1      # Single chunk only
            )
            
            logger.debug(f"Ultra-fast retrieval: {len(chunks)} chunk")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in ultra-fast retrieval: {e}")
            return []
    
    def _generate_ultra_fast_response(self, query: str, chunks: List[Any]) -> str:
        """Ultra-fast response with minimal context and very short prompt."""
        logger.debug("Generating ultra-fast response")
        
        try:
            # Minimal context (single chunk, truncated)
            context = ""
            if chunks:
                context = chunks[0].content[:400] + "..."  # Longer context for better results
                logger.debug(f"Using context: {len(context)} characters")
            
            # Ultra-short but effective prompt
            if context:
                prompt = f"""Answer using this context:

Context: {context}

Question: {query}

Answer:"""
            else:
                prompt = f"""Q: {query}
A:"""
            
            logger.debug(f"Ultra-short prompt: {len(prompt)} chars")
            
            # Generate very short response
            response = self.llm.invoke(prompt)
            
            logger.debug(f"Ultra-fast response: {len(response)} chars")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in ultra-fast response: {e}")
            return f"Error: {str(e)}"
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return minimal tool info."""
        return [
            {
                "name": "ultra_fast_retriever",
                "description": "Ultra-fast single-chunk retrieval",
                "parameters": {"query": "string"}
            }
        ]
