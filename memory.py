# Conversational memory and context management for the Agentic RAG Chatbot

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import os
from pathlib import Path

from models import ChatMessage, ChatSession
from config import settings
from utils import generate_id

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Represents a memory item."""
    id: str
    content: str
    memory_type: str  # "fact", "preference", "context", "summary"
    importance: float  # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int
    metadata: Dict[str, Any]


class ConversationalMemory:
    """Manages conversational memory and context."""
    
    def __init__(self, db_path: str = "memory.db"):
        """Initialize the memory system."""
        self.db_path = db_path
        self.max_memory_items = 1000
        self.memory_decay_days = 30
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for memory storage."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Create indexes
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items (memory_type)
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_items (importance)
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages (session_id)
            """)
            
            self.conn.commit()
            logger.info("Memory database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")
            raise
    
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5, 
                   metadata: Dict[str, Any] = None) -> str:
        """Add a new memory item."""
        try:
            memory_id = generate_id()
            now = datetime.now()
            
            memory_item = MemoryItem(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                created_at=now,
                last_accessed=now,
                access_count=0,
                metadata=metadata or {}
            )
            
            self.conn.execute("""
                INSERT INTO memory_items 
                (id, content, memory_type, importance, created_at, last_accessed, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_item.id,
                memory_item.content,
                memory_item.memory_type,
                memory_item.importance,
                memory_item.created_at.isoformat(),
                memory_item.last_accessed.isoformat(),
                memory_item.access_count,
                json.dumps(memory_item.metadata)
            ))
            
            self.conn.commit()
            logger.info(f"Added memory item: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    def get_relevant_memories(self, query: str, memory_type: str = None, 
                            limit: int = 10) -> List[MemoryItem]:
        """Get memories relevant to a query."""
        try:
            # Simple keyword-based relevance (can be enhanced with embeddings)
            query_words = set(query.lower().split())
            
            # Build query
            sql = """
                SELECT * FROM memory_items 
                WHERE 1=1
            """
            params = []
            
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
            params.append(limit * 2)  # Get more for filtering
            
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()
            
            # Filter by relevance
            relevant_memories = []
            for row in rows:
                memory = self._row_to_memory_item(row)
                
                # Simple relevance check
                content_words = set(memory.content.lower().split())
                relevance_score = len(query_words.intersection(content_words))
                
                if relevance_score > 0:
                    memory.metadata["relevance_score"] = relevance_score
                    relevant_memories.append(memory)
            
            # Sort by relevance and importance
            relevant_memories.sort(
                key=lambda x: (x.metadata.get("relevance_score", 0), x.importance),
                reverse=True
            )
            
            # Update access count and last accessed
            for memory in relevant_memories[:limit]:
                self._update_memory_access(memory.id)
            
            return relevant_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    def update_memory_importance(self, memory_id: str, importance: float):
        """Update the importance of a memory item."""
        try:
            self.conn.execute("""
                UPDATE memory_items 
                SET importance = ? 
                WHERE id = ?
            """, (importance, memory_id))
            
            self.conn.commit()
            logger.info(f"Updated memory importance: {memory_id}")
            
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            raise
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        try:
            cursor = self.conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            self.conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted memory: {memory_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    
    def cleanup_old_memories(self, days_threshold: int = None):
        """Clean up old, low-importance memories."""
        try:
            if days_threshold is None:
                days_threshold = self.memory_decay_days
            
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Delete old, low-importance memories
            cursor = self.conn.execute("""
                DELETE FROM memory_items 
                WHERE created_at < ? AND importance < 0.3
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old memories")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    AVG(importance) as avg_importance,
                    COUNT(CASE WHEN memory_type = 'fact' THEN 1 END) as fact_count,
                    COUNT(CASE WHEN memory_type = 'preference' THEN 1 END) as preference_count,
                    COUNT(CASE WHEN memory_type = 'context' THEN 1 END) as context_count,
                    COUNT(CASE WHEN memory_type = 'summary' THEN 1 END) as summary_count
                FROM memory_items
            """)
            
            row = cursor.fetchone()
            
            return {
                "total_memories": row["total_memories"],
                "average_importance": row["avg_importance"],
                "fact_memories": row["fact_count"],
                "preference_memories": row["preference_count"],
                "context_memories": row["context_count"],
                "summary_memories": row["summary_count"]
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def _update_memory_access(self, memory_id: str):
        """Update memory access information."""
        try:
            self.conn.execute("""
                UPDATE memory_items 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), memory_id))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating memory access: {e}")
    
    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convert database row to MemoryItem."""
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )


class SessionManager:
    """Manages chat sessions and their messages."""
    
    def __init__(self, memory: ConversationalMemory):
        """Initialize session manager."""
        self.memory = memory
        self.active_sessions: Dict[str, ChatSession] = {}
    
    def create_session(self, session_id: str = None) -> ChatSession:
        """Create a new chat session."""
        if session_id is None:
            session_id = generate_id()
        
        session = ChatSession(
            id=session_id,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        # Store in database
        self._store_session(session)
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from database
        session = self._load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    def add_message(self, session_id: str, message: ChatMessage):
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        session.messages.append(message)
        session.updated_at = datetime.now()
        session.metadata["message_count"] = len(session.messages)
        
        # Store in database
        self._store_message(message)
        
        # Extract and store important information as memories
        self._extract_memories_from_message(message)
    
    def get_session_history(self, session_id: str, limit: int = None) -> List[ChatMessage]:
        """Get session message history."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Delete from database
            self.memory.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self.memory.conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            self.memory.conn.commit()
            
            logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def _store_session(self, session: ChatSession):
        """Store session in database."""
        try:
            self.memory.conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, created_at, updated_at, message_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.id,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                len(session.messages),
                json.dumps(session.metadata)
            ))
            
            self.memory.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing session: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from database."""
        try:
            cursor = self.memory.conn.execute("""
                SELECT * FROM sessions WHERE id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Load messages
            messages_cursor = self.memory.conn.execute("""
                SELECT * FROM session_messages 
                WHERE session_id = ? 
                ORDER BY timestamp
            """, (session_id,))
            
            messages = []
            for msg_row in messages_cursor.fetchall():
                message = ChatMessage(
                    id=msg_row["id"],
                    content=msg_row["content"],
                    role=msg_row["role"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    metadata=json.loads(msg_row["metadata"]) if msg_row["metadata"] else {}
                )
                messages.append(message)
            
            session = ChatSession(
                id=row["id"],
                messages=messages,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None
    
    def _store_message(self, message: ChatMessage):
        """Store message in database."""
        try:
            self.memory.conn.execute("""
                INSERT OR REPLACE INTO session_messages 
                (id, session_id, content, role, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.session_id if hasattr(message, 'session_id') else None,
                message.content,
                message.role,
                message.timestamp.isoformat(),
                json.dumps(message.metadata)
            ))
            
            self.memory.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing message: {e}")
    
    def _extract_memories_from_message(self, message: ChatMessage):
        """Extract important information from messages and store as memories."""
        try:
            if message.role == "user":
                # Extract user preferences and facts
                content = message.content.lower()
                
                # Look for preference indicators
                if any(word in content for word in ["prefer", "like", "dislike", "want", "need"]):
                    self.memory.add_memory(
                        content=message.content,
                        memory_type="preference",
                        importance=0.7,
                        metadata={"extracted_from": message.id}
                    )
                
                # Look for factual statements
                if any(word in content for word in ["is", "are", "was", "were", "will be", "has", "have"]):
                    self.memory.add_memory(
                        content=message.content,
                        memory_type="fact",
                        importance=0.6,
                        metadata={"extracted_from": message.id}
                    )
            
            elif message.role == "assistant":
                # Extract context and summaries
                if len(message.content) > 100:  # Longer responses might contain summaries
                    self.memory.add_memory(
                        content=message.content[:500],  # Limit length
                        memory_type="context",
                        importance=0.5,
                        metadata={"extracted_from": message.id}
                    )
            
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")


class ContextManager:
    """Manages conversation context and memory integration."""
    
    def __init__(self, memory: ConversationalMemory, session_manager: SessionManager):
        """Initialize context manager."""
        self.memory = memory
        self.session_manager = session_manager
    
    def get_context_for_query(self, query: str, session_id: str = None, 
                            max_context_length: int = 2000) -> str:
        """Get relevant context for a query."""
        try:
            context_parts = []
            
            # Get relevant memories
            memories = self.memory.get_relevant_memories(query, limit=5)
            if memories:
                memory_context = "Relevant memories:\n"
                for memory in memories:
                    memory_context += f"- {memory.content}\n"
                context_parts.append(memory_context)
            
            # Get recent session history
            if session_id:
                recent_messages = self.session_manager.get_session_history(session_id, limit=10)
                if recent_messages:
                    history_context = "Recent conversation:\n"
                    for msg in recent_messages[-5:]:  # Last 5 messages
                        history_context += f"{msg.role}: {msg.content}\n"
                    context_parts.append(history_context)
            
            # Combine context
            full_context = "\n\n".join(context_parts)
            
            # Truncate if too long
            if len(full_context) > max_context_length:
                full_context = full_context[:max_context_length] + "..."
            
            return full_context
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def update_context_after_response(self, query: str, response: str, 
                                    session_id: str = None):
        """Update context after generating a response."""
        try:
            # Store important information from the interaction
            if session_id:
                # Add user query as context memory
                self.memory.add_memory(
                    content=f"User asked: {query}",
                    memory_type="context",
                    importance=0.6,
                    metadata={"session_id": session_id}
                )
                
                # Add assistant response as context memory
                if len(response) > 50:
                    self.memory.add_memory(
                        content=f"Assistant responded: {response[:200]}...",
                        memory_type="context",
                        importance=0.5,
                        metadata={"session_id": session_id}
                    )
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
