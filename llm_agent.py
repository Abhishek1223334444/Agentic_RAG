# LangGraph-based agent with multi-step reasoning capabilities

import logging
import re
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json

from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from models import ChatMessage, AgentState, QueryRequest, QueryResponse, DocumentChunk
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
    document_ids: Optional[List[str]]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    is_complete: bool
    session_id: str
    response: Optional[str]
    tools_executed: List[str]  # Track which tools have been executed
    needs_retrieval: bool
    needs_summarization: bool
    needs_comparison: bool


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
        """Build the LangGraph workflow with proper routing and termination."""
        try:
            # Create the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("agent", self._agent_node)
            workflow.add_node("retriever", self._retriever_node)
            workflow.add_node("summarizer", self._summarizer_node)
            workflow.add_node("comparator", self._comparator_node)
            workflow.add_node("finalize", self._finalize_node)
            
            # Add conditional routing from agent
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "retriever": "retriever",
                    "summarizer": "summarizer",
                    "comparator": "comparator",
                    "finalize": "finalize",
                    "end": END
                }
            )
            
            # All tool nodes return to agent for next decision
            workflow.add_edge("retriever", "agent")
            workflow.add_edge("summarizer", "agent")
            workflow.add_edge("comparator", "agent")
            
            # Finalize node goes to END
            workflow.add_edge("finalize", END)
            
            # Set entry point
            workflow.set_entry_point("agent")
            
            # Compile the graph with increased recursion limit
            self.graph = workflow.compile()
            
            logger.info("LangGraph workflow built successfully with conditional routing")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent reasoning node."""
        try:
            # Check if we should stop
            if state["iteration_count"] >= state["max_iterations"]:
                logger.warning(f"Max iterations ({state['max_iterations']}) reached, finalizing")
                state["is_complete"] = True
                return state
            
            # Check if already complete
            if state["is_complete"]:
                return state
            
            # Get the latest message
            messages = state["messages"]
            if not messages:
                state["is_complete"] = True
                return state
            
            latest_message = messages[-1]
            
            # Initialize tools_executed if not present
            if "tools_executed" not in state:
                state["tools_executed"] = []
            
            # On first iteration, analyze the query
            if state["iteration_count"] == 0 and isinstance(latest_message, HumanMessage):
                query = latest_message.content
                state["current_query"] = query
                
                # Analyze query to determine tools needed
                tool_decision = self._analyze_query(query)
                state["needs_retrieval"] = tool_decision["needs_retrieval"]
                state["needs_summarization"] = tool_decision["needs_summarization"]
                state["needs_comparison"] = tool_decision["needs_comparison"]
                
                logger.info(f"Query analysis: retrieval={tool_decision['needs_retrieval']}, "
                          f"summarization={tool_decision['needs_summarization']}, "
                          f"comparison={tool_decision['needs_comparison']}")
            
            # Increment iteration count
            state["iteration_count"] += 1
            logger.debug(f"Agent node iteration {state['iteration_count']}")
            
            # Check if we have enough information to respond
            if self._has_sufficient_information(state) and state["iteration_count"] > 1:
                logger.info("Sufficient information gathered, ready to finalize")
                state["is_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent node: {e}")
            state["is_complete"] = True
            state["response"] = f"Error processing query: {str(e)}"
            return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decision node - determines next step in the workflow."""
        try:
            logger.debug(f"Decision node: iteration={state.get('iteration_count', 0)}, is_complete={state.get('is_complete', False)}")
            
            # If complete, go to finalize
            if state.get("is_complete", False):
                logger.debug("State is complete, routing to finalize")
                return "finalize"
            
            # Check max iterations
            if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
                logger.warning("Max iterations reached, routing to finalize")
                return "finalize"
            
            # Determine which tool to execute
            tools_executed = state.get("tools_executed", [])
            logger.debug(f"Tools executed so far: {tools_executed}")
            
            # Priority 1: retrieval first (if needed and not executed)
            needs_retrieval = state.get("needs_retrieval", False)
            if needs_retrieval and "retriever" not in tools_executed:
                logger.info("Decision: Routing to retriever node")
                return "retriever"
            
            # Priority 2: summarization (if needed and not executed)
            needs_summarization = state.get("needs_summarization", False)
            if needs_summarization and "summarizer" not in tools_executed:
                # Only if we have content to summarize
                retrieved_chunks = state.get("retrieved_chunks", [])
                if retrieved_chunks or len(state.get("messages", [])) > 1:
                    logger.info("Decision: Routing to summarizer node")
                    return "summarizer"
            
            # Priority 3: comparison (if needed and not executed)
            needs_comparison = state.get("needs_comparison", False)
            if needs_comparison and "comparator" not in tools_executed:
                # Only if we have enough chunks for comparison
                retrieved_chunks = state.get("retrieved_chunks", [])
                if len(retrieved_chunks) >= 2:
                    logger.info("Decision: Routing to comparator node")
                    return "comparator"
            
            # If we have retrieved chunks, we can finalize
            if self._has_sufficient_information(state):
                logger.info("Decision: Sufficient information available, routing to finalize")
                return "finalize"
            
            # If no tools needed and we have a query, generate direct response
            if not needs_retrieval and not needs_summarization and not needs_comparison:
                logger.info("Decision: No tools needed, routing to finalize")
                return "finalize"
            
            # Default: finalize to prevent infinite loops
            logger.warning("Decision: No clear next step, routing to finalize to prevent infinite loop")
            return "finalize"
            
        except Exception as e:
            logger.error(f"Error in decision node (routing): {e}", exc_info=True)
            return "finalize"
    
    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine which tools are needed."""
        query_lower = query.lower()
        
        # Default to retrieval for most queries (RAG-first approach)
        # Only skip retrieval for very specific non-document queries
        skip_retrieval_keywords = ["hello", "hi", "thanks", "thank you", "bye", "goodbye"]
        needs_retrieval = not any(keyword in query_lower for keyword in skip_retrieval_keywords)
        
        # Check for specific retrieval keywords (enhances confidence)
        strong_retrieval_keywords = [
            "find", "search", "what", "where", "when", "how", "explain", 
            "tell me about", "describe", "define", "what is", "what are"
        ]
        if any(keyword in query_lower for keyword in strong_retrieval_keywords):
            needs_retrieval = True
        
        needs_summarization = any(keyword in query_lower for keyword in [
            "summarize", "summary", "brief", "overview", "main points"
        ])
        
        needs_comparison = any(keyword in query_lower for keyword in [
            "compare", "difference", "similar", "contrast", "versus", "vs"
        ])
        
        needs_voice = any(keyword in query_lower for keyword in [
            "speak", "voice", "audio", "listen", "hear"
        ])
        
        logger.info(f"Query analysis: retrieval={needs_retrieval}, summarization={needs_summarization}, comparison={needs_comparison}")
        
        return {
            "needs_retrieval": needs_retrieval,
            "needs_summarization": needs_summarization,
            "needs_comparison": needs_comparison,
            "needs_voice": needs_voice
        }
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever tool node - retrieves relevant document chunks."""
        try:
            logger.info("Retriever node executing...")
            
            # Ensure tools_executed list exists
            if "tools_executed" not in state:
                state["tools_executed"] = []
            
            # Mark retriever as executed to prevent loops
            if "retriever" not in state["tools_executed"]:
                state["tools_executed"].append("retriever")
                logger.info("Marked retriever as executed")
            
            # Ensure retrieved_chunks list exists
            if "retrieved_chunks" not in state:
                state["retrieved_chunks"] = []
            if "document_ids" not in state:
                state["document_ids"] = []
            
            query = state.get("current_query", "")
            if not query:
                logger.warning("No query found in state for retriever node")
                error_message = AIMessage(content="Retrieval failed: No query available")
                state["messages"].append(error_message)
                return state
            
            logger.info(f"Executing retriever tool for query: {query[:50]}...")
            
            # Execute retriever tool
            result = self.tool_registry.execute_tool(
                "retriever_tool",
                query=query,
                max_chunks=settings.max_context_chunks,
                threshold=0.3,  # Lower threshold to avoid missing relevant chunks
                document_ids=state.get("document_ids") or None,
            )
            
            if result.get("success", False):
                chunks = result.get("chunks", [])
                
                # Convert chunks to dict format for state
                state["retrieved_chunks"] = [chunk.dict() if hasattr(chunk, 'dict') else chunk for chunk in chunks]
                
                # Add tool call to messages
                tool_message = AIMessage(content=f"Retrieved {len(chunks)} relevant chunks for query: {query}")
                state["messages"].append(tool_message)
                
                logger.info(f"Retriever node completed: Retrieved {len(chunks)} chunks")
            else:
                # Even on failure, ensure retrieved_chunks is set (empty list)
                state["retrieved_chunks"] = []
                error_msg = result.get('error', 'Unknown error')
                error_message = AIMessage(content=f"Retrieval failed: {error_msg}")
                state["messages"].append(error_message)
                logger.warning(f"Retriever node failed: {error_msg}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retriever node: {e}", exc_info=True)
            error_message = AIMessage(content=f"Retriever error: {str(e)}")
            state["messages"].append(error_message)
            
            # Ensure state is properly initialized even on error
            if "tools_executed" not in state:
                state["tools_executed"] = []
            if "retriever" not in state["tools_executed"]:
                state["tools_executed"].append("retriever")
            if "retrieved_chunks" not in state:
                state["retrieved_chunks"] = []
            
            return state
    
    def _summarizer_node(self, state: AgentState) -> AgentState:
        """Summarizer tool node."""
        try:
            # Mark summarizer as executed
            if "tools_executed" not in state:
                state["tools_executed"] = []
            if "summarizer" not in state["tools_executed"]:
                state["tools_executed"].append("summarizer")
            
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
            # Mark as executed even on error
            if "tools_executed" not in state:
                state["tools_executed"] = []
            if "summarizer" not in state["tools_executed"]:
                state["tools_executed"].append("summarizer")
            return state
    
    def _comparator_node(self, state: AgentState) -> AgentState:
        """Comparator tool node."""
        try:
            # Mark comparator as executed
            if "tools_executed" not in state:
                state["tools_executed"] = []
            if "comparator" not in state["tools_executed"]:
                state["tools_executed"].append("comparator")
            
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
            # Mark as executed even on error
            if "tools_executed" not in state:
                state["tools_executed"] = []
            if "comparator" not in state["tools_executed"]:
                state["tools_executed"].append("comparator")
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
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize node - generates the final response and marks as complete."""
        try:
            logger.info("Finalize node executing...")
            
            # Check if response already exists
            if state.get("response") is not None:
                logger.info("Response already exists, skipping generation")
                state["is_complete"] = True
                return state
            
            # Get retrieved chunks
            retrieved_chunks = state.get("retrieved_chunks", [])
            query = state.get("current_query", "")
            
            logger.info(f"Finalize node: {len(retrieved_chunks)} chunks available, query: {query[:50]}...")
            
            # Generate response based on whether we have chunks
            if retrieved_chunks and len(retrieved_chunks) > 0:
                logger.info("Generating response from retrieved chunks")
                response = self._generate_final_response(state)
            else:
                logger.info("No chunks available, generating LLM response with disclaimer")
                response = self._generate_response(query, state)
            
            state["response"] = response
            state["is_complete"] = True
            
            logger.info(f"Finalize node completed: Response length={len(response)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in finalize node: {e}", exc_info=True)
            state["response"] = f"Error generating response: {str(e)}"
            state["is_complete"] = True
            return state
    
    def _has_sufficient_information(self, state: AgentState) -> bool:
        """Check if we have sufficient information to generate a response."""
        # Simple heuristic: if we have any retrieved chunks, we can answer from documents
        return len(state["retrieved_chunks"]) > 0
    
    def _generate_response(self, query: str, state: AgentState) -> str:
        """
        Generate a response when no document context is available.
        
        When no chunks are found, we still allow the LLM to answer,
        but with a clear disclaimer that the answer is not from the uploaded documents.
        """
        logger.warning("No retrieved chunks available; generating LLM answer with disclaimer")
        try:
            # Generate answer using LLM
            prompt = f"""You are a helpful AI assistant. Answer the following question based on your knowledge:

Question: {query}

Provide a clear and helpful response."""
            
            llm_answer = self.llm.invoke(prompt)
            
            # Prefix with disclaimer
            return f"The query is not there in the context but the answer is: {llm_answer}"
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"The query is not there in the context but the answer is: I encountered an error while processing your query: {str(e)}"
    
    def _generate_final_response(self, state: AgentState) -> str:
        """
        Generate the final response using retrieved information.
        
        Uses LLM with strict context enforcement to ensure answers
        are based only on the uploaded documents.
        """
        logger.info("Generating final response using LLM with retrieved chunks")
        try:
            chunks = state["retrieved_chunks"]
            query = state.get("current_query", "")
            
            if not chunks:
                logger.warning("No retrieved chunks available in _generate_final_response")
                return (
                    "I couldn't find any information about this in the uploaded documents. "
                    "Please check if the PDF or other files actually contain the answer."
                )

            # Deduplicate chunk contents (the same PDF might be uploaded multiple times)
            seen_contents = set()
            unique_contents: List[str] = []
            for chunk in chunks:
                text = (chunk.get("content") or "").strip()
                if text and text not in seen_contents:
                    seen_contents.add(text)
                    unique_contents.append(text)

            if not unique_contents:
                logger.warning("All retrieved chunks had empty content")
                return (
                    "I couldn't find any information about this in the uploaded documents. "
                    "Please check if the PDF or other files actually contain the answer."
                )

            # Prepare context from top chunks (limit to avoid token limits)
            max_context_chunks = min(len(unique_contents), getattr(settings, "max_context_chunks", 3))
            context_parts = []
            for i, content in enumerate(unique_contents[:max_context_chunks]):
                context_parts.append(f"Source {i+1}:\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response with strict RAG prompt
            prompt = f"""You are a strict RAG assistant. You MUST answer using ONLY the exact information provided in the context below.
You are NOT allowed to use outside knowledge or to guess.
When possible, use the exact text from the context. If you need to paraphrase, stay very close to the original wording.

Question: {query}

Context:
{context}

Instructions:
- Use ONLY the information that appears in the context above.
- If the context contains an exact definition or sentence that answers the question, use that information.
- You may paraphrase slightly for clarity, but stay very close to the original wording from the context.
- If the context does not contain enough information to fully answer, say: "Based on the uploaded documents, [partial answer]. The documents do not contain complete information about this topic."

Answer:"""
            
            logger.debug(f"Generated prompt for final response: {len(prompt)} characters")
            response = self.llm.invoke(prompt)
            
            logger.info(f"Generated final response: {len(response)} characters")
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating final response: {e}", exc_info=True)
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
                document_ids=request.document_ids,
                tool_calls=[],
                iteration_count=0,
                max_iterations=min(settings.max_iterations, 10),  # Cap at 10 to prevent recursion issues
                is_complete=False,
                session_id=request.session_id or generate_id(),
                response=None,
                tools_executed=[],
                needs_retrieval=False,
                needs_summarization=False,
                needs_comparison=False
            )
            
            # Run the graph with increased recursion limit
            config = {"recursion_limit": 50}  # Increased from default 25
            final_state = self.graph.invoke(initial_state, config=config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare source chunks (convert dicts back to models)
            source_chunks = []
            try:
                raw_chunks = final_state.get("retrieved_chunks", []) or []
                max_sources = getattr(settings, "max_context_chunks", 3)
                for chunk_dict in raw_chunks[:max_sources]:
                    # chunk_dict was created via chunk.dict(), so it should map cleanly
                    source_chunks.append(DocumentChunk(**chunk_dict))
            except Exception as e:
                logger.error(f"Error converting retrieved_chunks to DocumentChunk models: {e}")
                source_chunks = []

            # Create response
            response = QueryResponse(
                answer=final_state["response"] or "I couldn't generate a response.",
                sources=source_chunks,
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
