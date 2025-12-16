# Agent tools for the Agentic RAG Chatbot

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from models import DocumentChunk, ChatMessage
from embed_store import EmbeddingStore

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for the agent."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for the tool."""
        pass


class RetrieverTool(BaseTool):
    """Tool for retrieving relevant document chunks."""
    
    def __init__(self, embedding_store: EmbeddingStore):
        super().__init__(
            name="retriever_tool",
            description="Retrieve relevant document chunks based on a query"
        )
        self.embedding_store = embedding_store
    
    def execute(self, query: str, max_chunks: int = 5, threshold: float = 0.7, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the retrieval tool."""
        logger.info(f"Executing retriever tool: query='{query[:50]}...', max_chunks={max_chunks}, threshold={threshold}")
        try:
            logger.debug(f"Searching for relevant chunks with query: {query}")

            # Use similarity_search so threshold is applied inside the store
            chunks = self.embedding_store.similarity_search(
                query=query,
                threshold=threshold,
                max_results=max_chunks,
                document_ids=document_ids
            )
            logger.info(f"Retriever returned {len(chunks)} chunks after applying threshold {threshold}")
            
            # Prepare response
            response = {
                "success": True,
                "chunks": chunks,
                "count": len(chunks),
                "query": query,
                "threshold": threshold
            }
            
            logger.info(f"Retriever tool completed successfully: {len(chunks)} chunks returned")
            return response
            
        except Exception as e:
            logger.error(f"Error in retriever tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "count": 0
            }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Maximum number of chunks to retrieve",
                    "default": 5
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold for filtering results",
                    "default": 0.3
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of document IDs to restrict retrieval"
                }
            },
            "required": ["query"]
        }


class SummarizerTool(BaseTool):
    """Tool for summarizing document chunks."""
    
    def __init__(self, llm_client):
        super().__init__(
            name="summarizer_tool",
            description="Summarize document chunks or text content"
        )
        self.llm_client = llm_client
    
    def execute(self, content: str, max_length: int = 200, style: str = "concise") -> Dict[str, Any]:
        """Execute the summarization tool."""
        logger.info(f"Executing summarizer tool: content_length={len(content)}, max_length={max_length}, style={style}")
        try:
            logger.debug(f"Content to summarize: {content[:200]}...")
            
            # Prepare prompt based on style
            if style == "concise":
                prompt = f"Provide a concise summary of the following text in {max_length} words or less:\n\n{content}"
            elif style == "detailed":
                prompt = f"Provide a detailed summary of the following text, highlighting key points:\n\n{content}"
            elif style == "bullet_points":
                prompt = f"Summarize the following text as bullet points:\n\n{content}"
            else:
                prompt = f"Summarize the following text:\n\n{content}"
            
            logger.debug(f"Generated prompt for style '{style}': {prompt[:200]}...")
            
            # Generate summary using LLM
            logger.info("Generating summary using LLM")
            summary = self.llm_client.generate_response(prompt)
            
            response = {
                "success": True,
                "summary": summary,
                "original_length": len(content.split()),
                "summary_length": len(summary.split()),
                "style": style,
                "max_length": max_length
            }
            
            logger.info(f"Summarizer tool completed successfully: {len(summary.split())} words generated")
            return response
            
        except Exception as e:
            logger.error(f"Error in summarizer tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "",
                "original_length": 0,
                "summary_length": 0
            }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of summary in words",
                    "default": 200
                },
                "style": {
                    "type": "string",
                    "description": "Summary style",
                    "enum": ["concise", "detailed", "bullet_points"],
                    "default": "concise"
                }
            },
            "required": ["content"]
        }


class ComparatorTool(BaseTool):
    """Tool for comparing document chunks or concepts."""
    
    def __init__(self, llm_client):
        super().__init__(
            name="comparator_tool",
            description="Compare document chunks or concepts to find similarities and differences"
        )
        self.llm_client = llm_client
    
    def execute(self, content1: str, content2: str, comparison_type: str = "comprehensive") -> Dict[str, Any]:
        """Execute the comparison tool."""
        logger.info(f"Executing comparator tool: content1_length={len(content1)}, content2_length={len(content2)}, type={comparison_type}")
        try:
            logger.debug(f"Content 1: {content1[:100]}...")
            logger.debug(f"Content 2: {content2[:100]}...")
            
            # Prepare comparison prompt
            if comparison_type == "comprehensive":
                prompt = f"""Compare the following two pieces of content and identify:
1. Key similarities
2. Key differences
3. Main themes in each
4. Overall assessment

Content 1:
{content1}

Content 2:
{content2}

Provide a structured comparison."""
            
            elif comparison_type == "similarities":
                prompt = f"""Find the main similarities between these two pieces of content:

Content 1:
{content1}

Content 2:
{content2}

Focus on common themes, ideas, and concepts."""
            
            elif comparison_type == "differences":
                prompt = f"""Find the main differences between these two pieces of content:

Content 1:
{content1}

Content 2:
{content2}

Focus on contrasting ideas, approaches, and concepts."""
            
            else:
                prompt = f"""Compare these two pieces of content:

Content 1:
{content1}

Content 2:
{content2}"""
            
            logger.debug(f"Generated comparison prompt for type '{comparison_type}': {prompt[:200]}...")
            
            # Generate comparison using LLM
            logger.info("Generating comparison using LLM")
            comparison = self.llm_client.generate_response(prompt)
            
            response = {
                "success": True,
                "comparison": comparison,
                "content1_length": len(content1.split()),
                "content2_length": len(content2.split()),
                "comparison_type": comparison_type
            }
            
            logger.info(f"Comparator tool completed successfully: {len(comparison)} characters generated")
            return response
            
        except Exception as e:
            logger.error(f"Error in comparator tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "comparison": "",
                "content1_length": 0,
                "content2_length": 0
            }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content1": {
                    "type": "string",
                    "description": "First content to compare"
                },
                "content2": {
                    "type": "string",
                    "description": "Second content to compare"
                },
                "comparison_type": {
                    "type": "string",
                    "description": "Type of comparison to perform",
                    "enum": ["comprehensive", "similarities", "differences"],
                    "default": "comprehensive"
                }
            },
            "required": ["content1", "content2"]
        }


class VoiceTool(BaseTool):
    """Tool for voice processing (speech-to-text and text-to-speech)."""
    
    def __init__(self):
        super().__init__(
            name="voice_tool",
            description="Process voice input (speech-to-text) and generate voice output (text-to-speech)"
        )
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Whisper and gTTS models."""
        try:
            import whisper
            from gtts import gTTS
            import io
            
            # Initialize Whisper model
            self.whisper_model = whisper.load_model("base")
            
            logger.info("Voice tool initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing voice tool: {e}")
            self.whisper_model = None
    
    def execute(self, operation: str, data: Any = None, language: str = "en") -> Dict[str, Any]:
        """Execute voice operations."""
        logger.info(f"Executing voice tool: operation={operation}, language={language}, data_type={type(data).__name__}")
        try:
            if operation == "speech_to_text":
                logger.info("Processing speech-to-text conversion")
                return self._speech_to_text(data, language)
            elif operation == "text_to_speech":
                logger.info("Processing text-to-speech conversion")
                return self._text_to_speech(data, language)
            else:
                logger.warning(f"Unknown voice operation: {operation}")
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Error in voice tool: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _speech_to_text(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """Convert speech to text using Whisper."""
        logger.info(f"Converting speech to text: audio_size={len(audio_data)} bytes, language={language}")
        try:
            if not self.whisper_model:
                logger.error("Whisper model not initialized")
                return {
                    "success": False,
                    "error": "Whisper model not initialized"
                }
            
            import tempfile
            import os
            
            # Save audio data to temporary file
            logger.debug("Saving audio data to temporary file")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            logger.debug(f"Temporary file created: {temp_file_path}")
            
            try:
                # Transcribe audio
                logger.info("Transcribing audio using Whisper")
                result = self.whisper_model.transcribe(temp_file_path, language=language)
                text = result["text"].strip()
                
                response = {
                    "success": True,
                    "text": text,
                    "language": language,
                    "confidence": result.get("segments", [{}])[0].get("avg_logprob", 0) if result.get("segments") else 0
                }
                
                logger.info(f"Speech-to-text completed successfully: {len(text)} characters transcribed")
                return response
                
            finally:
                # Clean up temporary file
                logger.debug(f"Cleaning up temporary file: {temp_file_path}")
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def _text_to_speech(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Convert text to speech using gTTS."""
        logger.info(f"Converting text to speech: text_length={len(text)} characters, language={language}")
        try:
            from gtts import gTTS
            import io
            
            logger.debug(f"Text to synthesize: {text[:100]}...")
            
            # Generate speech
            logger.info("Generating speech using gTTS")
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Convert to bytes
            logger.debug("Converting speech to bytes")
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            
            response = {
                "success": True,
                "audio_data": audio_data,
                "text": text,
                "language": language,
                "audio_size": len(audio_data)
            }
            
            logger.info(f"Text-to-speech completed successfully: {len(audio_data)} bytes generated")
            return response
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Voice operation to perform",
                    "enum": ["speech_to_text", "text_to_speech"]
                },
                "data": {
                    "description": "Audio data for speech-to-text or text for text-to-speech"
                },
                "language": {
                    "type": "string",
                    "description": "Language code",
                    "default": "en"
                }
            },
            "required": ["operation", "data"]
        }


class ToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        logger.info(f"Registering tool: {tool.name}")
        self.tools[tool.name] = tool
        logger.info(f"Successfully registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        logger.debug(f"Retrieving tool: {name}")
        tool = self.tools.get(name)
        if tool:
            logger.debug(f"Tool {name} found")
        else:
            logger.warning(f"Tool {name} not found")
        return tool
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        logger.debug(f"Retrieving all tools: {len(self.tools)} available")
        return self.tools.copy()
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        logger.debug("Retrieving tool schemas")
        schemas = [tool.get_schema() for tool in self.tools.values()]
        logger.debug(f"Retrieved {len(schemas)} tool schemas")
        return schemas
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        logger.info(f"Executing tool: {name}")
        tool = self.get_tool(name)
        if not tool:
            logger.error(f"Tool '{name}' not found")
            return {
                "success": False,
                "error": f"Tool '{name}' not found"
            }
        
        try:
            logger.debug(f"Executing tool {name} with parameters: {list(kwargs.keys())}")
            result = tool.execute(**kwargs)
            logger.info(f"Tool {name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
