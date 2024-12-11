from rag_system import RAGPrototype
import os
import logging
from tqdm import tqdm
import time
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_test_questions() -> List[str]:
    """Return a list of test questions focused on current condo documents."""
    return [
        # General Overview
        "What are the governing documents for condos?",
        "What requirements should owners be mindful of when it comes to AGMs?",
        
        # Meeting Specific
        "Who attended the meeting that is covered in the minutes document?",
        "What important decisions or actions were proposed in the management report and covered in the meeting?",
        
        # Owner's Guide
        "What is the Tribunal process and who is involved?",
        "What are the Owner's maintenance and repair responsibilities mentioned?"
    ]

def format_response(response: dict, question: str) -> str:
    """Format the response with clear separation and source attribution."""
    formatted = f"\n{'='*80}\n"
    formatted += f"Question: {question}\n"
    formatted += f"{'='*80}\n\n"
    formatted += f"Answer: {response['answer']}\n"
    
    if response.get('sources'):
        formatted += f"\nSources Used:\n{'-'*40}\n"
        for i, source in enumerate(response['sources'], 1):
            # Truncate long sources for readability
            truncated_source = source[:300] + "..." if len(source) > 300 else source
            formatted += f"\nSource {i}:\n{truncated_source}\n"
            formatted += f"{'-'*40}\n"
    
    return formatted

def main():
    # Get API key from environment
    llama_parse_api_key = os.getenv('LLAMA_PARSE_API_KEY')
    if not llama_parse_api_key:
        logger.error("LLAMA_PARSE_API_KEY not found in environment variables")
        return
    
    # Create pdfs directory if it doesn't exist
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Check if there are any PDFs
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        logger.info("Please add PDF files to the 'pdfs' directory before running the test.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files:")
    for file in pdf_files:
        logger.info(f"  - {file}")
    
    try:
        # Initialize RAG system with API key
        logger.info("Initializing RAG system with Llama 3...")
        rag = RAGPrototype(
            pdf_directory=pdf_dir, 
            llama_parse_api_key=llama_parse_api_key
        )
        
        # Load documents
        logger.info("Loading and processing documents with LlamaParse...")
        rag.load_documents()
        
        # Get test questions
        questions = get_test_questions()
        logger.info(f"Preparing to process {len(questions)} test queries...")
        
        # Process each question with progress bar
        with tqdm(total=len(questions), desc="Processing queries") as pbar:
            for question in questions:
                try:
                    # Show current question being processed
                    pbar.set_description(f"Processing: {question[:50]}...")
                    
                    # Get response
                    response = rag.query(question, include_sources=True)
                    
                    # Print formatted response
                    print(format_response(response, question))
                    
                    # Small delay for readability
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {str(e)}")
                finally:
                    pbar.update(1)
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise
    finally:
        logger.info("Test run completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        logger.info("Note: Partial results may have been saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Program terminated.")