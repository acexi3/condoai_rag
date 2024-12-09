"""
RAG (Retrieval Augmented Generation) System for Condo Document Analysis
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from llama_parse import LlamaParse
import pdfplumber
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import time
import re

def setup_logging():
    """Configure logging with reduced verbosity and separate file logs."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create timestamped log filename
    log_filename = f'logs/rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Custom filter for console output
    class VerbosityFilter(logging.Filter):
        def filter(self, record):
            return not any(x in record.msg.lower() 
                         for x in ['embedding', 'http', 'token', 'chunk'])
    
    # Create handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    
    # Add filter to console handler
    verbosity_filter = VerbosityFilter()
    console_handler.addFilter(verbosity_filter)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            console_handler
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class DocumentStructure:
    """Handles document structure recognition and content extraction."""
    
    def __init__(self, custom_patterns: Optional[Dict] = None):
        # Default patterns
        self.default_patterns = {
            'action_items': r'(?:^|\n)ACTION:\s*(.*?)(?=\n\n|\Z)',
            'new_business': r'(?:^|\n)NEW BUSINESS\s*(.*?)(?=\n\n|\Z)',  # Removed number dependency
            'call_to_order': r'(?:^|\n)CALL TO ORDER\s*(.*?)(?=\n\n|\Z)',
            'correspondence': r'(?:^|\n)CORRESPONDENCE\s*(.*?)(?=\n\n|\Z)',
            'meeting_times': {
                'start': r'(?:start|begin|order).*?(\d{1,2}:\d{2}\s*(?:am|pm))',
                'end': r'(?:end|adjourn|close).*?(\d{1,2}:\d{2}\s*(?:am|pm))'
            },
            'motions': r'(?:^|\n)(?:MOTION|On a MOTION).*?(?:carried|defeated)',
            'attendees': r'(?:^|\n)(?:Present|Attendance):\s*(.*?)(?=\n\n|\Z)',
            'management_report': r'(?:^|\n)MANAGEMENT REPORT\s*(.*?)(?=\n\n|\Z)',
            'financial_report': r'(?:^|\n)FINANCIAL REPORT\s*(.*?)(?=\n\n|\Z)',
            'variance_report': r'(?:^|\n)VARIANCE REPORT\s*(.*?)(?=\n\n|\Z)',
            'review_of_unaudited_financial_statements': r'(?:^|\n)REVIEW OF UNAUDITED FINANCIAL STATEMENTS\s*(.*?)(?=\n\n|\Z)',
            'review_and_approval_of_meeting_minutes': r'(?:^|\n)REVIEW AND APPROVAL OF MEETING MINUTES\s*(.*?)(?=\n\n|\Z)',
        }
        
        # Update with any custom patterns
        self.patterns = self.default_patterns.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)
    
    def add_pattern(self, name: str, pattern: str):
        """Add a new pattern or update existing one."""
        self.patterns[name] = pattern
    
    def remove_pattern(self, name: str):
        """Remove a pattern by name."""
        if name in self.patterns:
            del self.patterns[name]
    
    def get_patterns(self) -> Dict:
        """Get current patterns."""
        return self.patterns
    
    def extract_content(self, text: str) -> Dict[str, List[str]]:
        """Extract structured content from document text."""
        content = {}
        
        for key, pattern in self.patterns.items():
            if isinstance(pattern, str):
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                content[key] = [m.group(1).strip() if '(' in pattern else m.group().strip() 
                              for m in matches]
            elif isinstance(pattern, dict):  # For meeting_times
                times = {}
                for time_type, time_pattern in pattern.items():
                    match = re.search(time_pattern, text, re.IGNORECASE)
                    if match:
                        times[time_type] = match.group(1)
                content[key] = times
        
        return content

class DocumentClassifier:
    """Classifies documents and determines relevance to queries."""
    
    def __init__(self):
        self.document_types = {
            'minutes': {
                'patterns': [r'minutes', r'agenda', r'meeting', r'report'],
                'keywords': ['action', 'motion', 'quorum', 'approve', 'ratify', 'attendance', 'regret', 'invitation']
            },
            'governance': {
                'patterns': [r'by-?laws?', r'declaration', r'rules', r'policy'],
                'keywords': ['rule', 'regulation', 'policy', 'bylaws', 'declaration']
            },
            'maintenance': {
                'patterns': [r'maintenance', r'repair'],
                'keywords': ['fix', 'responsibility', 'obligation', 'repair', 'maintenance']
            },
            'financial': {
                'patterns': [r'financial', r'budget', r'statement', r'arrears', r'audit'],
                'keywords': ['cost', 'expense', 'fee', 'payment', 'arrears', 'audit']
            }
        }
    
    def classify_document(self, text: str) -> List[str]:
        """Determine document types based on content."""
        doc_types = []
        
        for doc_type, criteria in self.document_types.items():
            if any(re.search(pattern, text, re.IGNORECASE) 
                  for pattern in criteria['patterns']):
                doc_types.append(doc_type)
                continue
            
            if any(keyword in text.lower() 
                  for keyword in criteria['keywords']):
                doc_types.append(doc_type)
        
        return list(set(doc_types))
    
class PDFExtractor:
    """Handles PDF document extraction with enhanced metadata."""
    
    def __init__(self, llama_parse_api_key: Optional[str] = None, custom_patterns: Optional[Dict] = None):
        self.llama_parse_api_key = llama_parse_api_key
        self.llama_parser = LlamaParse(api_key=llama_parse_api_key) if llama_parse_api_key else None
        self.doc_structure = DocumentStructure(custom_patterns)  # Pass custom patterns
       
    def extract_from_pdf(self, file_path: str) -> List[Document]:
        """Extract text and metadata from PDF."""
        try:
            if self.llama_parser:
                return self._extract_with_llama_parse(file_path)
            return self._extract_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {str(e)}")
            raise
    
    def _extract_with_llama_parse(self, file_path: str) -> List[Document]:
        """Extract using LlamaParse with enhanced metadata."""
        documents = self.llama_parser.load_data(file_path)
        
        # Add structure recognition
        for doc in documents:
            structured_content = self.doc_structure.extract_content(doc.page_content)
            doc.metadata['structured_content'] = structured_content
            doc.metadata['extraction_method'] = 'llama_parse'
        
        return documents
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Document]:
        """Extract using pdfplumber with enhanced metadata."""
        documents = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Add structure recognition
                    structured_content = self.doc_structure.extract_content(text)
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': file_path,
                            'page': page_num,
                            'structured_content': structured_content,
                            'extraction_method': 'pdfplumber'
                        }
                    )
                    documents.append(doc)
        
        return documents
    
class ProgressEmbeddings(OllamaEmbeddings):
    """Adds progress tracking to embedding generation."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with progress bar."""
        logger.info(f"Creating embeddings for {len(texts)} text chunks...")
        
        with tqdm(total=len(texts), desc="Creating embeddings") as pbar:
            results = []
            for text in texts:
                results.append(self.embed_query(text))
                pbar.update(1)
                time.sleep(0.1)  # Rate limiting
        
        return results
    
class RAGPrototype:
    """Enhanced RAG system with structure awareness."""
    
    def __init__(self, 
                 pdf_directory: str, 
                 llama_parse_api_key: Optional[str] = None,
                 custom_patterns: Optional[Dict] = None):
        """
        Initialize RAG system with optional custom patterns.
        
        Args:
            pdf_directory: Directory containing PDF files
            llama_parse_api_key: Optional API key for LlamaParse
            custom_patterns: Optional dictionary of custom regex patterns
        """
        self.pdf_directory = pdf_directory
        self.pdf_extractor = PDFExtractor(
            llama_parse_api_key=llama_parse_api_key,
            custom_patterns=custom_patterns
        )
        self.doc_classifier = DocumentClassifier()
        self.documents = []
        self.vectorstore = None
        self.structured_content = {}  # Store structured content by document
        
        # Initialize LLM and embeddings with Llama3
        self.llm = OllamaLLM(
            model="llama3",
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.embeddings = ProgressEmbeddings(model="llama3")
    
    def load_documents(self):
        """Load and process documents with enhanced structure recognition."""
        logger.info("Loading documents...")
        
        for pdf_file in os.listdir(self.pdf_directory):
            if pdf_file.endswith('.pdf'):
                file_path = os.path.join(self.pdf_directory, pdf_file)
                try:
                    # Extract documents with structure
                    docs = self.pdf_extractor.extract_from_pdf(file_path)
                    
                    # Store structured content
                    for doc in docs:
                        # Classify document
                        doc_types = self.doc_classifier.classify_document(doc.page_content)
                        doc.metadata['types'] = doc_types
                        
                        # Store structured content separately for quick access
                        if 'structured_content' in doc.metadata:
                            self.structured_content[file_path] = doc.metadata['structured_content']
                    
                    self.documents.extend(docs)
                    logger.info(f"Processed {pdf_file} - Types: {doc_types}")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        self._create_vectorstore()
    
    def _create_vectorstore(self):
        """Create vector store from processed documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(self.documents)
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
    
    def query(self, question: str) -> dict:
        """Process query with structure awareness."""
        try:
            # Determine relevant document types
            relevant_types = self.doc_classifier.classify_document(question)
            logger.info(f"Query relevant to document types: {relevant_types}")
            
            # Check structured content first for specific information
            structured_answer = self._check_structured_content(question)
            
            # Get relevant chunks
            k = self._determine_chunk_count(question)
            docs = self._get_relevant_chunks(question, k, relevant_types)
            
            # Generate response
            qa_chain = self._create_qa_chain()
            context = self._combine_context(docs, structured_answer)
            response = qa_chain({"question": question, "context": context})
            
            return {
                "answer": response["result"],
                "sources": docs,
                "structured_content": structured_answer if structured_answer else None
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _determine_chunk_count(self, question: str) -> int:
        """Determine number of chunks based on question complexity."""
        words = len(question.split())
        if words > 20 or '?' in question:
            return 6
        return 4
    
    def _get_relevant_chunks(self, question: str, k: int, relevant_types: List[str]) -> List[Document]:
        """Get relevant chunks with type filtering."""
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Filter by relevant types if specified
        if relevant_types:
            docs = [
                doc for doc in docs 
                if any(t in doc.metadata.get('types', []) for t in relevant_types)
            ]
        
        return docs
    
    def _check_structured_content(self, question: str) -> Optional[str]:
        """Check structured content for specific information."""
        structured_info = []
        
        # Define question-to-pattern mapping
        pattern_mapping = {
            'time': ['meeting_times'],
            'when': ['meeting_times'],
            'action': ['action_items'],
            'motion': ['motions'],
            'attend': ['attendees'],
            'present': ['attendees'],
            'business': ['new_business', 'old_business'],
            'management': ['management_report'],
            'financial': ['financial_report']
        }
        
        # Find relevant patterns for the question
        relevant_patterns = []
        for keyword, patterns in pattern_mapping.items():
            if keyword in question.lower():
                relevant_patterns.extend(patterns)
        
        # Check structured content for relevant patterns
        if relevant_patterns:
            for doc_path, content in self.structured_content.items():
                for pattern in relevant_patterns:
                    if pattern in content and content[pattern]:
                        doc_name = os.path.basename(doc_path)
                        structured_info.append(f"{pattern} from {doc_name}:")
                        if isinstance(content[pattern], dict):
                            for key, value in content[pattern].items():
                                structured_info.append(f"  {key}: {value}")
                        else:
                            for item in content[pattern]:
                                structured_info.append(f"  - {item}")
        
        return "\n".join(structured_info) if structured_info else None
    
    def _combine_context(self, docs: List[Document], structured_info: Optional[str]) -> str:
        """Combine retrieved documents with structured information."""
        context_parts = []
        
        if structured_info:
            context_parts.append("Structured Information:\n" + structured_info)
        
        context_parts.append("Retrieved Content:\n" + 
                           "\n".join(doc.page_content for doc in docs))
        
        return "\n\n".join(context_parts)
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create QA chain with custom prompt."""
        template = """
        Answer the question based only on the provided context. If you cannot answer the question based on the context, say "I cannot answer this based on the provided documents."

        Context: {context}

        Question: {question}

        Answer: Let me analyze the provided documents and give you a structured answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )