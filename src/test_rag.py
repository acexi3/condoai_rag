import os 
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from rag_system import RAGPrototype

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_patterns():
    """Test specific patterns against document content."""
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize RAG with test patterns
    test_patterns = {
        'test_pattern': r'(?:^|\n)TEST\s*SECTION:\s*(.*?)(?=\n\n|\Z)',
        'custom_dates': r'(?:^|\n)DATE:\s*(\d{1,2}/\d{1,2}/\d{4})',
    }

    rag = RAGPrototype(
        pdf_directory=os.path.join(os.path.dirname(__file__), "..", "pdfs"),
        llama_parse_api_key=os.getenv('LLAMA_PARSE_API_KEY'),
        custom_patterns=test_patterns
    )

    try:
        # Load documents
        rag.load_documents()
        
        # Test each pattern type
        pattern_tests = {
            'default_patterns': [
                "What were the action items?",
                "When did the meeting start and end?",
                "List all motions from the meeting.",
                "Who attended the meeting?",
                "Who is the manager of the corporation?",
                "What is the name or number of the corporation?",
            ],
            'custom_patterns': [
                "What special items were discussed?",
                "What building maintenance was mentioned?",
            ],
            'test_patterns': [
                "What was in the test section?",
                "What dates were mentioned?",
            ]
        }

        # Store results
        results = {
            'timestamp': datetime.now().isoformat(),
            'pattern_results': {}
        }

        # Test each pattern category
        for category, questions in pattern_tests.items():
            logger.info(f"\nTesting {category}...")
            category_results = []
            
            for question in questions:
                logger.info(f"\nQuestion: {question}")
                try:
                    result = rag.query(question)
                    
                    # Check if structured content was found
                    structured_found = bool(result.get('structured_content'))
                    
                    category_results.append({
                        'question': question,
                        'answer': result['answer'],
                        'structured_content_found': structured_found,
                        'sources': [
                            {
                                'content': doc.page_content[:200] + '...',  # First 200 chars
                                'metadata': doc.metadata
                            } for doc in result.get('sources', [])
                        ]
                    })
                    
                    logger.info(f"Structured content found: {structured_found}")
                    
                except Exception as e:
                    logger.error(f"Error testing pattern: {str(e)}")
                    category_results.append({
                        'question': question,
                        'error': str(e)
                    })
            
            results['pattern_results'][category] = category_results

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"pattern_test_results_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\nResults saved to: {results_file}")

    except Exception as e:
        logger.error(f"Error during pattern test: {str(e)}")
        raise

if __name__ == "__main__":
    test_patterns()