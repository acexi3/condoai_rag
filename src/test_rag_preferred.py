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
        "What are all the governing documents for condos?",
        "What are the AGM requirements?",
        "What are reserve funds used for in condos?",
        "What are the reserve fund requirements?",

        # Financial Information
        "What specific financial information and budget items are discussed across all the documents? Please cite each document separately.",
        "Can you please provide a summary of the review of the unaudited financial statements in the meeting?",

        # Governance
        "What are the mandatory requirements versus optional guidelines for condo governance? Please distinguish between requirements and recommendations.",
        
        # Meeting Specific
        "Who attended the meeting that is referencedd in the minutes document?",
        "What decisions were made by the board in the meeting?",
        "What topics did the manager discuss in the management report?",
        "Does the agenda match the topics discussed in the management report and minutes?",
        "What is the date of the next meeting?",
        "Who were the guests at the meeting?",
        "Who is the manager of the condo?",
        "What are the action items from the meeting?",
        "Was there any new business discussed at the meeting?",
        "What time did the meeting start and end?",

        # Owner's Guide
        "What is the Tribunal process and who is involved?",
        "Which maintenance and repair responsibilities are the owner's and which are the condo's responsibility?",
        "What are owner's insurance responsibilities and best practices?",
        
        # Table-Specific (since we now handle tables better)
        "What financial information or budget items are discussed in these documents?"
    ]

def format_response(
        response: dict, 
        question: str
    ) -> str:
    """Format the response with clear separation and source attribution."""
    formatted = f"\n{'='*80}\n"
    formatted += f"Question: {question}\n"
    formatted += f"{'='*80}\n\n"
    formatted += f"Answer: {response['answer']}\n"
    
    if response.get('sources'):
        formatted += f"\nSources Used:\n{'-'*40}\n"
        for i, source in enumerate(response['sources'], 1):
            # Include content type in source attribution
            content_type = source.metadata.get('type', 'unknown')
            page_num = source.metadata.get('page', 'unknown')
            
            formatted += f"\nSource {i} (Type: {content_type}, Page: {page_num}):\n"
            # Use page_content instead of string conversion
            content = source.page_content
            truncated_content = content[:300] + "..." if len(content) > 300 else content
            formatted += f"{truncated_content}\n"
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
        rag = RAGPrototype(pdf_directory=pdf_dir, llama_parse_api_key=llama_parse_api_key)
        
        # Load documents
        logger.info("Loading and processing documents...")
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