# LangGraph-based agent with multi-step reasoning capabilities

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json

from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from models import ChatMessage, AgentState, QueryRequest, QueryResponse
from tools import ToolRegistry, RetrieverTool, SummarizerTool, ComparatorTool, VoiceTool
from embed_store import EmbeddingStore
from config import settings
from utils import generate_id

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_query: str
    retrieved_chunks: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    is_complete: bool
    session_id: str
    response: Optional[str]


class LLMAgent:
    """LangGraph-based agent with multi-step reasoning capabilities."""
    
    def __init__(self, embedding_store: EmbeddingStore):
        """Initialize the agent."""
        self.embedding_store = embedding_store
        self.llm = None
        self.tool_registry = ToolRegistry()
        self.graph = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, tools, and graph."""
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.temperature
            )
            
            # Initialize tools
            self._initialize_tools()
            
            # Build the graph
            self._build_graph()
            
            logger.info("LLM Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLM Agent: {e}")
            raise
    
    def _initialize_tools(self):
        """Initialize and register agent tools."""
        try:
            # Create tools
            retriever_tool = RetrieverTool(self.embedding_store)
            summarizer_tool = SummarizerTool(self.llm)
            comparator_tool = ComparatorTool(self.llm)
            voice_tool = VoiceTool()
            
            # Register tools
            self.tool_registry.register_tool(retriever_tool)
            self.tool_registry.register_tool(summarizer_tool)
            self.tool_registry.register_tool(comparator_tool)
            self.tool_registry.register_tool(voice_tool)
            
            logger.info(f"Initialized {len(self.tool_registry.get_all_tools())} tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            raise
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        try:
            # Create the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("agent", self._agent_node)
            workflow.add_node("tools", self._tool_node)
            workflow.add_node("retriever", self._retriever_node)
            workflow.add_node("summarizer", self._summarizer_node)
            workflow.add_node("comparator", self._comparator_node)
            workflow.add_node("voice", self._voice_node)
            
            # Add edges
            workflow.add_edge("agent", "tools")
            workflow.add_edge("tools", "agent")
            workflow.add_edge("retriever", "agent")
            workflow.add_edge("summarizer", "agent")
            workflow.add_edge("comparator", "agent")
            workflow.add_edge("voice", "agent")
            
            # Set entry point
            workflow.set_entry_point("agent")
            
            # Compile the graph
            self.graph = workflow.compile()
            
            logger.info("LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent reasoning node."""
        try:
            # Check if we should continue
            if state["iteration_count"] >= state["max_iterations"]:
                state["is_complete"] = True
                return state
            
            # Get the latest message
            messages = state["messages"]
            if not messages:
                return state
            
            latest_message = messages[-1]
            
            # Determine next action based on message content
            if isinstance(latest_message, HumanMessage):
                query = latest_message.content
                state["current_query"] = query
                
                # Analyze query to determine tools needed
                tool_decision = self._analyze_query(query)
                
                if tool_decision["needs_retrieval"]:
                    # Go to retriever
                    return state
                elif tool_decision["needs_summarization"]:
                    # Go to summarizer
                    return state
                elif tool_decision["needs_comparison"]:
                    # Go to comparator
                    return state
                elif tool_decision["needs_voice"]:
                    # Go to voice
                    return state
                else:
                    # Generate direct response
                    response = self._generate_response(query, state)
                    state["response"] = response
                    state["is_complete"] = True
            
            elif isinstance(latest_message, AIMessage):
                # Agent has completed a tool call, continue reasoning
                state["iteration_count"] += 1
                
                # Check if we have enough information to respond
                if self._has_sufficient_information(state):
                    response = self._generate_final_response(state)
                    state["response"] = response
                    state["is_complete"] = True
                else:
                    # Continue with more tool calls
                    pass
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent node: {e}")
            state["is_complete"] = True
            state["response"] = f"Error processing query: {str(e)}"
            return state
    
    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine which tools are needed."""
        query_lower = query.lower()
        
        # Simple keyword-based analysis (can be enhanced with more sophisticated NLP)
        needs_retrieval = any(keyword in query_lower for keyword in [
            "find", "search", "what", "where", "when", "how", "explain", "tell me about"
        ])
        
        needs_summarization = any(keyword in query_lower for keyword in [
            "summarize", "summary", "brief", "overview", "main points"
        ])
        
        needs_comparison = any(keyword in query_lower for keyword in [
            "compare", "difference", "similar", "contrast", "versus", "vs"
        ])
        
        needs_voice = any(keyword in query_lower for keyword in [
            "speak", "voice", "audio", "listen", "hear"
        ])
        
        return {
            "needs_retrieval": needs_retrieval,
            "needs_summarization": needs_summarization,
            "needs_comparison": needs_comparison,
            "needs_voice": needs_voice
        }
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever tool node."""
        try:
            query = state["current_query"]
            
            # Execute retriever tool
            result = self.tool_registry.execute_tool("retriever_tool", query=query)
            
            if result["success"]:
                chunks = result["chunks"]
                state["retrieved_chunks"] = [chunk.dict() for chunk in chunks]
                
                # Add tool call to messages
                tool_message = AIMessage(content=f"Retrieved {len(chunks)} relevant chunks for query: {query}")
                state["messages"].append(tool_message)
                
                logger.info(f"Retrieved {len(chunks)} chunks")
            else:
                error_message = AIMessage(content=f"Retrieval failed: {result.get('error', 'Unknown error')}")
                state["messages"].append(error_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retriever node: {e}")
            error_message = AIMessage(content=f"Retriever error: {str(e)}")
            state["messages"].append(error_message)
            return state
    
    def _summarizer_node(self, state: AgentState) -> AgentState:
        """Summarizer tool node."""
        try:
            # Get content to summarize
            content = ""
            if state["retrieved_chunks"]:
                content = " ".join([chunk["content"] for chunk in state["retrieved_chunks"]])
            else:
                content = state["current_query"]
            
            # Execute summarizer tool
            result = self.tool_registry.execute_tool("summarizer_tool", content=content)
            
            if result["success"]:
                summary = result["summary"]
                tool_message = AIMessage(content=f"Summary: {summary}")
                state["messages"].append(tool_message)
                
                logger.info("Generated summary successfully")
            else:
                error_message = AIMessage(content=f"Summarization failed: {result.get('error', 'Unknown error')}")
                state["messages"].append(error_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in summarizer node: {e}")
            error_message = AIMessage(content=f"Summarizer error: {str(e)}")
            state["messages"].append(error_message)
            return state
    
    def _comparator_node(self, state: AgentState) -> AgentState:
        """Comparator tool node."""
        try:
            # For now, use retrieved chunks for comparison
            # This can be enhanced to handle more complex comparison scenarios
            chunks = state["retrieved_chunks"]
            
            if len(chunks) >= 2:
                content1 = chunks[0]["content"]
                content2 = chunks[1]["content"]
                
                # Execute comparator tool
                result = self.tool_registry.execute_tool("comparator_tool", 
                                                       content1=content1, 
                                                       content2=content2)
                
                if result["success"]:
                    comparison = result["comparison"]
                    tool_message = AIMessage(content=f"Comparison: {comparison}")
                    state["messages"].append(tool_message)
                    
                    logger.info("Generated comparison successfully")
                else:
                    error_message = AIMessage(content=f"Comparison failed: {result.get('error', 'Unknown error')}")
                    state["messages"].append(error_message)
            else:
                error_message = AIMessage(content="Need at least 2 chunks for comparison")
                state["messages"].append(error_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in comparator node: {e}")
            error_message = AIMessage(content=f"Comparator error: {str(e)}")
            state["messages"].append(error_message)
            return state
    
    def _voice_node(self, state: AgentState) -> AgentState:
        """Voice tool node."""
        try:
            # For text-to-speech, use the current query or response
            text = state["current_query"]
            
            # Execute voice tool
            result = self.tool_registry.execute_tool("voice_tool", 
                                                    operation="text_to_speech", 
                                                    data=text)
            
            if result["success"]:
                tool_message = AIMessage(content=f"Generated audio for: {text[:100]}...")
                state["messages"].append(tool_message)
                
                logger.info("Generated audio successfully")
            else:
                error_message = AIMessage(content=f"Voice generation failed: {result.get('error', 'Unknown error')}")
                state["messages"].append(error_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in voice node: {e}")
            error_message = AIMessage(content=f"Voice error: {str(e)}")
            state["messages"].append(error_message)
            return state
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """Generic tool execution node."""
        # This can be used for additional tool processing
        return state
    
    def _has_sufficient_information(self, state: AgentState) -> bool:
        """Check if we have sufficient information to generate a response."""
        # Simple heuristic: if we have retrieved chunks or completed tool calls
        return len(state["retrieved_chunks"]) > 0 or len(state["tool_calls"]) > 0
    
    def _generate_response(self, query: str, state: AgentState) -> str:
        """Generate a direct response without tools."""
        try:
            prompt = f"""You are a helpful AI assistant. Answer the following question based on your knowledge:

Question: {query}

Provide a clear and helpful response."""
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def _generate_final_response(self, state: AgentState) -> str:
        """Generate the final response using retrieved information."""
        logger.info("Generating final response using retrieved information")
        try:
            query = state["current_query"]
            chunks = state["retrieved_chunks"]
            
            logger.info(f"Using {len(chunks)} retrieved chunks for final response")
            
            # Prepare context from retrieved chunks
            context = ""
            if chunks:
                context = "\n\n".join([f"Source {i+1}: {chunk['content']}" 
                                     for i, chunk in enumerate(chunks[:3])])  # Use top 3 chunks
                logger.debug(f"Prepared context from {min(len(chunks), 3)} chunks")
            
            # Generate response with context
            prompt = f"""You are a helpful AI assistant. Answer the following question using the provided context:

Question: {query}

Context:
{context}

Provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so clearly."""
            
            logger.debug(f"Generated final prompt: {prompt[:300]}...")
            response = self.llm.invoke(prompt)
            
            logger.info(f"Generated final response of {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using the agent."""
        try:
            start_time = datetime.now()
            
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=request.query)],
                current_query=request.query,
                retrieved_chunks=[],
                tool_calls=[],
                iteration_count=0,
                max_iterations=settings.max_iterations,
                is_complete=False,
                session_id=request.session_id or generate_id(),
                response=None
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = QueryResponse(
                answer=final_state["response"] or "I couldn't generate a response.",
                sources=[],  # Convert chunks to DocumentChunk objects if needed
                session_id=final_state["session_id"],
                query_time=processing_time,
                metadata={
                    "iterations": final_state["iteration_count"],
                    "tools_used": len(final_state["tool_calls"]),
                    "chunks_retrieved": len(final_state["retrieved_chunks"])
                }
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with {final_state['iteration_count']} iterations")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                answer=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                session_id=request.session_id or generate_id(),
                query_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        logger.info("Retrieving available tools")
        tools = self.tool_registry.get_tool_schemas()
        logger.info(f"Retrieved {len(tools)} available tools")
        return tools
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session (placeholder for future implementation)."""
        logger.info(f"Reset session request for: {session_id}")
        # This can be implemented to clear session-specific state
        logger.warning("Session reset not yet implemented")
        return True
