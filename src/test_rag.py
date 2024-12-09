import os 
import logging
from rag_system import RAGPrototype

def test_rag():
    # Optional: Define custom patterns for testing
    custom_patterns = {
        'special_items': r'(?:^|\n)(?:SPECIAL|ADDITIONAL)\s*ITEMS:\s*(.*?)(?=\n\n|\Z)',
    }

    # Initialize RAG
    rag = RAGPrototype(
        pdf_directory=os.path.join(os.path.dirname(__file__), "..", "pdfs"),
        llama_parse_api_key=None,  # Add your key if using LlamaParse
        custom_patterns=custom_patterns
    )

    try:
        # Load documents
        rag.load_documents()
        
        # Test queries
        test_questions = [
            "What were the action items from the meeting?",
            "When did the meeting start and end?",
            "What motions were passed?",
            "Who attended the meeting?",
            "What was discussed in new business?"
        ]

        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("=" * 80)
            
            result = rag.query(question)
            
            print("\nAnswer:", result["answer"])
            if result.get("structured_content"):
                print("\nStructured Content:", result["structured_content"])
            print("=" * 80)

    except Exception as e:
        logging.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_rag()