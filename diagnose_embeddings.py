#!/usr/bin/env python3
"""
Embedding Search Diagnostic Script
Tests the embedding search functionality to identify issues
"""

import requests
import json
from typing import List, Dict, Any

def test_embedding_search():
    """Test the embedding search functionality."""
    print("🔍 Testing Embedding Search Functionality")
    print("=" * 50)
    
    # Test queries with different thresholds
    test_queries = [
        "Abhishek Shaw",
        "Siddhasiri Music Society", 
        "music society",
        "Abhishek",
        "society details"
    ]
    
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    
    for query in test_queries:
        print(f"\n📝 Testing Query: '{query}'")
        print("-" * 30)
        
        for threshold in thresholds:
            try:
                # Test with different thresholds
                response = requests.post(
                    "http://localhost:8000/chat/query",
                    json={
                        "query": query,
                        "session_id": f"test_{threshold}",
                        "max_chunks": 5
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    chunks_retrieved = data.get("metadata", {}).get("chunks_retrieved", 0)
                    context_length = data.get("metadata", {}).get("context_length", 0)
                    
                    print(f"   Threshold {threshold}: {chunks_retrieved} chunks, {context_length} chars")
                    
                    if chunks_retrieved > 0:
                        print(f"   ✅ Found chunks with threshold {threshold}")
                        print(f"   Answer: {data.get('answer', '')[:100]}...")
                        break
                else:
                    print(f"   ❌ Error {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")

def test_direct_search():
    """Test direct search without similarity threshold."""
    print(f"\n🔍 Testing Direct Search (No Threshold)")
    print("=" * 50)
    
    # Create a simple test script to test the embedding store directly
    test_script = """
import sys
sys.path.append('.')
from embed_store import EmbeddingStore
from config import settings

# Initialize embedding store
store = EmbeddingStore()

# Test queries
test_queries = [
    "Abhishek Shaw",
    "Siddhasiri Music Society",
    "music society",
    "Abhishek"
]

for query in test_queries:
    print(f"\\nQuery: {query}")
    
    # Test regular search
    chunks = store.search(query, max_results=5)
    print(f"  Regular search: {len(chunks)} chunks")
    
    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"    Chunk {i+1}: {chunk.content[:100]}...")
            print(f"    Similarity: {chunk.metadata.get('similarity_score', 'N/A')}")
    
    # Test similarity search with different thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        sim_chunks = store.similarity_search(query, threshold=threshold, max_results=3)
        print(f"  Similarity search (t={threshold}): {len(sim_chunks)} chunks")
"""
    
    with open("test_embedding_direct.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_embedding_direct.py")
    print("   Run: python test_embedding_direct.py")

def check_collection_details():
    """Check detailed collection information."""
    print(f"\n📊 Collection Details")
    print("=" * 30)
    
    try:
        # Get documents
        docs_response = requests.get("http://localhost:8000/documents")
        if docs_response.status_code == 200:
            documents = docs_response.json()
            print(f"Documents: {len(documents)}")
            
            for doc in documents:
                print(f"  📄 {doc['filename']}")
                print(f"     Type: {doc['document_type']}")
                print(f"     Chunks: {doc['chunks_count']}")
                print(f"     Content Length: {doc['content_length']}")
                
                # Get detailed document info
                doc_response = requests.get(f"http://localhost:8000/documents/{doc['id']}")
                if doc_response.status_code == 200:
                    doc_data = doc_response.json()
                    chunks = doc_data.get('chunks', [])
                    print(f"     Actual Chunks Retrieved: {len(chunks)}")
                    
                    if chunks:
                        print(f"     Sample Chunk: {chunks[0]['content'][:100]}...")
        
        # Get stats
        stats_response = requests.get("http://localhost:8000/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            collection_stats = stats.get('collection_stats', {})
            print(f"\nCollection Stats:")
            print(f"  Total Chunks: {collection_stats.get('total_chunks', 0)}")
            print(f"  Collection Name: {collection_stats.get('collection_name', 'N/A')}")
            print(f"  Embedding Model: {collection_stats.get('embedding_model', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error checking collection: {e}")

def main():
    """Main diagnostic function."""
    print("🔬 Embedding Search Diagnostic Tool")
    print("=" * 60)
    
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API is not running. Please start it first.")
            return
    except:
        print("❌ API is not running. Please start it first.")
        return
    
    print("✅ API is running")
    
    # Run diagnostics
    check_collection_details()
    test_embedding_search()
    test_direct_search()
    
    print(f"\n💡 Troubleshooting Tips:")
    print(f"   1. Check if sentence-transformers is working")
    print(f"   2. Verify ChromaDB collection has data")
    print(f"   3. Try lower similarity thresholds")
    print(f"   4. Check if document content was extracted properly")

if __name__ == "__main__":
    main()
