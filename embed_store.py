# ChromaDB-based embedding store for document storage and retrieval

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import numpy as np
from models import Document, DocumentChunk
from config import settings
from utils import generate_id, chunk_text, clean_text

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """ChromaDB-based embedding store for document storage and retrieval."""
    
    def __init__(self):
        """Initialize the embedding store."""
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.reranker = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ChromaDB client, collection, and models."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            
            # Initialize reranker (only if enabled for speed)
            if settings.enable_reranking:
                logger.info(f"Loading reranker model: {settings.reranker_model}")
                self.reranker = FlagReranker(settings.reranker_model, use_fp16=True)
                logger.info("Reranker model loaded successfully")
            else:
                logger.info("Reranking disabled for speed optimization")
                self.reranker = None
            
            logger.info("Embedding store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embedding store: {e}")
            raise
    
    def add_document(self, document: Document) -> str:
        """Add a document to the store."""
        try:
            # Process document into chunks
            chunks = self._create_chunks(document)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Prepare data for ChromaDB
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added document {document.id} with {len(chunks)} chunks")
            return document.id
            
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            raise
    
    def _create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create chunks from document content."""
        chunks = []
        chunk_texts = chunk_text(
            document.content,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        
        for i, chunk_content in enumerate(chunk_texts):
            chunk = DocumentChunk(
                chunk_id=generate_id(),
                document_id=document.id,
                content=clean_text(chunk_content),
                metadata={
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "document_type": document.document_type.value,
                    "upload_time": document.upload_time.isoformat()
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage."""
        metadata = chunk.metadata.copy()
        # Ensure all values are JSON serializable
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                metadata[key] = str(value)
        return metadata
    
    def search(self, query: str, max_results: int = None) -> List[DocumentChunk]:
        """Search for relevant chunks."""
        try:
            if max_results is None:
                max_results = settings.max_chunks
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results * 2,  # Get more results for reranking
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            chunks = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    chunk = DocumentChunk(
                        chunk_id=metadata.get("chunk_id", generate_id()),
                        document_id=metadata.get("document_id", ""),
                        content=doc,
                        metadata={
                            **metadata,
                            "similarity_score": 1 - distance,  # Convert distance to similarity
                            "rank": i + 1
                        }
                    )
                    chunks.append(chunk)
            
            # Rerank results if we have a reranker
            if self.reranker and len(chunks) > 1:
                chunks = self._rerank_chunks(query, chunks)
            
            # Return top results
            return chunks[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {e}")
            raise
    
    def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Rerank chunks using BGE reranker."""
        try:
            if len(chunks) <= 1:
                return chunks
            
            # Prepare query-document pairs for reranking
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Get reranking scores
            scores = self.reranker.compute_score(pairs)
            
            # Handle both single score and batch scores
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            # Update chunks with reranking scores and sort
            for chunk, score in zip(chunks, scores):
                chunk.metadata["rerank_score"] = float(score)
            
            # Sort by reranking score (descending)
            chunks.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
            
            # Update ranks
            for i, chunk in enumerate(chunks):
                chunk.metadata["final_rank"] = i + 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error reranking chunks: {e}")
            # Return original chunks if reranking fails
            return chunks
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific document."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results["documents"]:
                for doc, metadata in zip(results["documents"], results["metadatas"]):
                    chunk = DocumentChunk(
                        chunk_id=metadata.get("chunk_id", generate_id()),
                        document_id=document_id,
                        content=doc,
                        metadata=metadata
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["metadatas"]:
                chunk_ids = [metadata.get("chunk_id") for metadata in results["metadatas"]]
                # Remove None values
                chunk_ids = [cid for cid in chunk_ids if cid is not None]
                
                if chunk_ids:
                    self.collection.delete(ids=chunk_ids)
                    logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": settings.chroma_collection_name,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all data)."""
        try:
            self.client.delete_collection(settings.chroma_collection_name)
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def similarity_search(self, query: str, threshold: float = 0.7, max_results: int = None) -> List[DocumentChunk]:
        """Search with similarity threshold."""
        logger.info(f"Similarity search: query='{query[:50]}...', threshold={threshold}, max_results={max_results}")
        chunks = self.search(query)
        filtered_chunks = [chunk for chunk in chunks 
                if chunk.metadata.get("similarity_score", 0) >= threshold]
        
        # Apply max_results limit if specified
        if max_results is not None:
            filtered_chunks = filtered_chunks[:max_results]
        
        logger.info(f"Similarity search completed: {len(filtered_chunks)}/{len(chunks)} chunks above threshold")
        return filtered_chunks
    
    def get_related_chunks(self, chunk_id: str, max_results: int = 5) -> List[DocumentChunk]:
        """Get chunks related to a specific chunk."""
        logger.info(f"Getting related chunks for: {chunk_id}, max_results={max_results}")
        try:
            # Get the chunk
            logger.debug(f"Retrieving chunk: {chunk_id}")
            results = self.collection.get(
                ids=[chunk_id],
                include=["embeddings"]
            )
            
            if not results["embeddings"]:
                logger.warning(f"Chunk {chunk_id} not found")
                return []
            
            # Search for similar chunks
            logger.debug(f"Searching for similar chunks using embedding")
            similar_results = self.collection.query(
                query_embeddings=results["embeddings"],
                n_results=max_results + 1,  # +1 to exclude the original chunk
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            if similar_results["documents"] and similar_results["documents"][0]:
                logger.debug(f"Processing {len(similar_results['documents'][0])} similar chunks")
                for doc, metadata, distance in zip(
                    similar_results["documents"][0],
                    similar_results["metadatas"][0],
                    similar_results["distances"][0]
                ):
                    # Skip the original chunk
                    if metadata.get("chunk_id") == chunk_id:
                        continue
                    
                    chunk = DocumentChunk(
                        chunk_id=metadata.get("chunk_id", generate_id()),
                        document_id=metadata.get("document_id", ""),
                        content=doc,
                        metadata={
                            **metadata,
                            "similarity_score": 1 - distance,
                            "related_to": chunk_id
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} related chunks for {chunk_id}")
            return chunks[:max_results]
            
        except Exception as e:
            logger.error(f"Error getting related chunks for {chunk_id}: {e}")
            return []
