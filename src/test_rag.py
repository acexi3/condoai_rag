import os 
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables before any other imports
load_dotenv()

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

# Verify API key is loaded and log working directory
logger.info(f"Current working directory: {os.getcwd()}")
api_key = os.getenv('LLAMA_PARSE_API_KEY')
if not api_key:
    raise ValueError("LLAMA_PARSE_API_KEY not found in environment variables")
logger.info(f"LlamaParse API key loaded successfully: {api_key[:8]}...")

from rag_system import RAGPrototype

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
                 # General Overview
                "What are all the governing documents for condos?",
                "What are reserve funds used for in condos?",
                "What are the reserve fund requirements?",

                # Financial Information
                "What specific financial information and budget items are discussed across all the documents? Please cite each document separately.",
                "Can you please provide a summary of the review of the unaudited financial statements in the meeting?",

                # Governance
                "What are the mandatory requirements versus optional guidelines for condo governance, distinguishing between requirements and recommendations.",

                # Meeting Specific
                "When did the meeting start and end?",
                "Does the agenda match the topics discussed in the management report and minutes?",
                "What were the action items from this meeting?",
                "List all motions from the meeting.",
                "Who attended the meeting?",
                "Who is the manager of the corporation?",
                "What is the name or number of the corporation?",
            ],
            'custom_patterns': [
                "Were there any contracts mentioned in the meeting?",
                "Are there any owners in arrears currently?",
                "What was the status of the reserve fund?",
                "What was the status of building maintenance?",
            ],
            'test_patterns': [
                "What was in the test section?",
                "What dates were mentioned?",
            ]
        }

        # Store results
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'pattern_results': {}
        }
        
        # Store clean results (just Q&A)
        clean_results = {
            'timestamp': datetime.now().isoformat(),
            'questions_and_answers': []
        }

        # Test each pattern category
        for category, questions in pattern_tests.items():
            logger.info(f"\nTesting {category}...")
            category_results = []
            
            for question in questions:
                logger.info(f"\nQuestion: {question}")
                try:
                    result = rag.query(question)
                    
                    # Store detailed results
                    category_results.append({
                        'question': question,
                        'answer': result['answer'],
                        'structured_content_found': bool(result.get('structured_content')),
                        'sources': [
                            {
                                'content': doc.page_content[:200] + '...',  # First 200 chars
                                'metadata': doc.metadata
                            } for doc in result.get('sources', [])
                        ]
                    })
                    
                    # Store clean results
                    clean_results['questions_and_answers'].append({
                        'category': category,
                        'question': question,
                        'answer': result['answer']
                    })
                    
                    logger.info(f"Structured content found: {bool(result.get('structured_content'))}")
                    
                except Exception as e:
                    logger.error(f"Error testing pattern: {str(e)}")
                    error_result = {
                        'question': question,
                        'error': str(e)
                    }
                    category_results.append(error_result)
                    clean_results['questions_and_answers'].append({
                        'category': category,
                        'question': question,
                        'error': str(e)
                    })
            
            detailed_results['pattern_results'][category] = category_results

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_results_file = os.path.join(results_dir, f"pattern_test_detailed_{timestamp}.json")
        clean_results_file = os.path.join(results_dir, f"pattern_test_qa_{timestamp}.json")
        
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
        with open(clean_results_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\nDetailed results saved to: {detailed_results_file}")
        logger.info(f"Clean Q&A results saved to: {clean_results_file}")

    except Exception as e:
        logger.error(f"Error during pattern test: {str(e)}")
        raise

if __name__ == "__main__":
    test_patterns()