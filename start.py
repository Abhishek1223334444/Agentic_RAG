#!/usr/bin/env python3
"""
Startup script for the Agentic RAG Chatbot
This script helps users start the application with proper checks and setup.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import streamlit
        import chromadb
        import langchain
        import ollama
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def check_ollama():
    """Check if Ollama is running and has the required model."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama is not running")
            print("Please start Ollama: ollama serve")
            return False
        
        # Check if Mistral model is available
        models = response.json().get("models", [])
        mistral_models = [m for m in models if "mistral" in m.get("name", "").lower()]
        
        if not mistral_models:
            print("❌ Mistral model not found")
            print("Please install: ollama pull mistral:7b")
            return False
        
        print("✅ Ollama is running with Mistral model")
        return True
        
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Ollama")
        print("Please start Ollama: ollama serve")
        return False


def check_environment():
    """Check if environment file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Creating .env from template...")
        
        env_example = Path("env_example.txt")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from template")
        else:
            print("❌ env_example.txt not found")
            return False
    
    print("✅ Environment configuration found")
    return True


def start_backend():
    """Start the FastAPI backend."""
    print("\n🚀 Starting FastAPI backend...")
    try:
        # Start the backend process
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if it's running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ FastAPI backend is running on http://localhost:8000")
                return process
            else:
                print("❌ FastAPI backend failed to start")
                return None
        except requests.exceptions.RequestException:
            print("❌ FastAPI backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None


def start_frontend():
    """Start the Streamlit frontend."""
    print("\n🎨 Starting Streamlit frontend...")
    try:
        # Start the frontend process
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "ui/app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        print("✅ Streamlit frontend is starting on http://localhost:8501")
        return process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None


def main():
    """Main startup function."""
    print("🤖 Agentic RAG Chatbot Startup")
    print("=" * 40)
    
    # Run checks
    checks = [
        check_python_version(),
        check_dependencies(),
        check_environment(),
        check_ollama()
    ]
    
    if not all(checks):
        print("\n❌ Startup checks failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Start services
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend. Exiting.")
        sys.exit(1)
    
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend. Exiting.")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Application started successfully!")
    print("\n📱 Access the application:")
    print("   • Streamlit UI: http://localhost:8501")
    print("   • FastAPI Docs: http://localhost:8000/docs")
    print("   • API Health: http://localhost:8000/health")
    
    print("\n💡 Tips:")
    print("   • Upload documents through the Streamlit interface")
    print("   • Ask questions about your documents")
    print("   • Use voice features for hands-free interaction")
    
    print("\n⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        
        print("✅ Application stopped")


if __name__ == "__main__":
    main()
