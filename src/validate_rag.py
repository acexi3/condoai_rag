import os
import logging
from rag_system import RAGPrototype, DocumentStructure

def validate_system():
    """Validate RAG system components and configuration."""
    
    print("\n=== RAG System Validation ===\n")
    
    # 1. Check Ollama availability
    print("Checking Ollama service...")
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="llama3")
        llm.invoke("test")  # Simple test query
        print("✓ Ollama is running and Llama3 is available")
    except Exception as e:
        print("✗ Error with Ollama:", str(e))
        return False

    # 2. Check PDF directory
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "pdfs") 
    print(f"\nChecking PDF directory ({pdf_dir})...")
    if os.path.exists(pdf_dir):
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        print(f"✓ Found {len(pdfs)} PDF files")
        if not pdfs:
            print("! Warning: No PDF files found in directory")
    else:
        print("✗ PDF directory not found")
        return False

    # 3. Test pattern matching
    print("\nTesting pattern matching...")
    doc_structure = DocumentStructure()
    test_text = """
    ACTION ITEMS:
    1. Review budget
    
    NEW BUSINESS:
    Discussion about maintenance
    
    Present: John Doe, Jane Smith
    
    Meeting started at 2:00 PM
    Meeting ended at 4:00 PM
    """
    
    content = doc_structure.extract_content(test_text)
    patterns_found = [k for k, v in content.items() if v]
    print(f"✓ Pattern matching working - found {len(patterns_found)} patterns:")
    for pattern in patterns_found:
        print(f"  - {pattern}")

    # 4. Test basic RAG functionality
    print("\nTesting RAG initialization...")
    try:
        rag = RAGPrototype(
            pdf_directory=pdf_dir,
            custom_patterns={"test": r"test"}
        )
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print("✗ Error initializing RAG:", str(e))
        return False

    # 5. Check logging configuration
    print("\nChecking logging configuration...")
    if not os.path.exists('logs'):
        os.makedirs('logs')
    try:
        logging.info("Test log message")
        print("✓ Logging configured correctly")
    except Exception as e:
        print("✗ Error with logging:", str(e))
        return False

    print("\n=== Validation Complete ===")
    print("\nSystem is ready for testing!")
    return True

if __name__ == "__main__":
    if validate_system():
        print("\nYou can now run the test script!")
    else:
        print("\nPlease fix the issues before running tests.") 