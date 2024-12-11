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
from langchain_community.vectorstores.utils import filter_complex_metadata
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

def filter_complex_metadata(metadata: dict) -> dict:
    """Filter out complex metadata values, keeping only simple types."""
    filtered = {}
    for key, value in metadata.items():
        # Keep only simple types (str, int, float, bool)
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        elif isinstance(value, (list, dict)):
            # Convert complex types to string representation
            filtered[key] = str(value)
    return filtered

class DocumentStructure:
    """Handles document structure recognition and content extraction."""
    
    # Default patterns for standard meeting minutes sections
    DEFAULT_PATTERNS = {
        # action items
        'action_items': r'(?:^|\n)ACTION\s*:\s*(.*?)(?=\n\n|\Z)',
        'action_items_lower': r'(?:^|\n)action\s*:\s*(.*?)(?=\n\n|\Z)',
        'action_items_variation': r'(?:^|\n)Action\s*Items?\s*:\s*(.*?)(?=\n\n|\Z)',
    
        'new_business': r'(?:^|\n)NEW BUSINESS\s*(.*?)(?=\n\n|\Z)',
        'call_to_order': r'(?:^|\n)CALL TO ORDER\s*(.*?)(?=\n\n|\Z)',
        'correspondence': r'(?:^|\n)CORRESPONDENCE\s*(.*?)(?=\n\n|\Z)',
        'meeting_times': {
            'start': r'(?:start|begin|order).*?(\d{1,2}:\d{2}\s*(?:am|pm))',
            'end': r'(?:end|adjourn|close).*?(\d{1,2}:\d{2}\s*(?:am|pm))'
        },
        'motions': r'(?:^|\n)(?:MOTION|On a MOTION).*?(?:carried|defeated)',
        'attendees': r'(?:^|\n)(?:Present|Attendance):\s*(.*?)(?=\n\n|\Z)',
        'regrets': r'(?:^|\n)(?:Absent|Regret):\s*(.*?)(?=\n\n|\Z)',
        'management_report': r'(?:^|\n)MANAGEMENT REPORT\s*(.*?)(?=\n\n|\Z)',
        'financial_report': r'(?:^|\n)FINANCIAL REPORT\s*(.*?)(?=\n\n|\Z)',
        'statements': r'(?:^|\n)STATEMENTS\s*(.*?)(?=\n\n|\Z)',
        'variance_report': r'(?:^|\n)VARIANCE REPORT\s*(.*?)(?=\n\n|\Z)',
        'arrears_report': r'(?:^|\n)ARREARS REPORT\s*(.*?)(?=\n\n|\Z)',
        'review_of_unaudited_financial_statements': r'(?:^|\n)REVIEW OF UNAUDITED FINANCIAL STATEMENTS\s*(.*?)(?=\n\n|\Z)',
        'review_and_approval_of_meeting_minutes': r'(?:^|\n)REVIEW AND APPROVAL OF MEETING MINUTES\s*(.*?)(?=\n\n|\Z)',
        'items_for_discussion': r'(?:^|\n)ITEMS FOR DISCUSSION\s*(.*?)(?=\n\n|\Z)',
        'items_in_progress': r'(?:^|\n)ITEMS IN PROGRESS\s*(.*?)(?=\n\n|\Z)',
        'items_completed': r'(?:^|\n)ITEMS COMPLETED\s*(.*?)(?=\n\n|\Z)',
        'items_deferred': r'(?:^|\n)ITEMS DEFERRED\s*(.*?)(?=\n\n|\Z)',
        'date_of_next_meeting': r'(?:^|\n)DATE OF NEXT MEETING\s*(.*?)(?=\n\n|\Z)',
        'close_of_meeting': r'(?:^|\n)CLOSE OF MEETING\s*(.*?)(?=\n\n|\Z)',
    }

    # Additional custom patterns for specific document sections
    CUSTOM_PATTERNS = {
        'special_items': r'(?:^|\n)(?:SPECIAL|ADDITIONAL)\s*ITEMS:\s*(.*?)(?=\n\n|\Z)',
        'building_maintenance': r'(?:^|\n)BUILDING\s*MAINTENANCE:\s*(.*?)(?=\n\n|\Z)',
        'resident_complaints': r'(?:^|\n)RESIDENT\s*COMPLAINTS:\s*(.*?)(?=\n\n|\Z)',
        'upcoming_projects': r'(?:^|\n)UPCOMING\s*PROJECTS:\s*(.*?)(?=\n\n|\Z)',
    }
    
    def __init__(self, additional_patterns: Optional[Dict] = None):
        """
        Initialize document structure with patterns.
        
        Args:
            additional_patterns: Optional dictionary of additional patterns to include
        """
        # Combine default and custom patterns
        self.patterns = {**self.DEFAULT_PATTERNS, **self.CUSTOM_PATTERNS}
        
        # Add any additional patterns provided during initialization
        if additional_patterns:
            self.patterns.update(additional_patterns)
            
        logger.info("Active document patterns:")
        for pattern_name in self.patterns.keys():
            logger.info(f"  - {pattern_name}")
    
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
            try:
                if isinstance(pattern, str):
                    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                    content[key] = []
                    for m in matches:
                        try:
                            # Try to get group 1, fall back to group 0 if it doesn't exist
                            content[key].append(
                                m.group(1).strip() if m.groups() else m.group(0).strip()
                            )
                        except IndexError:
                            # If no groups exist, use the entire match
                            content[key].append(m.group(0).strip())
                elif isinstance(pattern, dict):  # For meeting_times
                    times = {}
                    for time_type, time_pattern in pattern.items():
                        match = re.search(time_pattern, text, re.IGNORECASE)
                        if match:
                            try:
                                times[time_type] = match.group(1)
                            except IndexError:
                                times[time_type] = match.group(0)
                    content[key] = times
            except Exception as e:
                logger.warning(f"Error extracting pattern '{key}': {str(e)}")
                content[key] = []
        
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
                logger.info(f"Using LlamaParse for {file_path}")
                return self._extract_with_llama_parse(file_path)
            logger.info(f"No LlamaParse available, using pdfplumber for {file_path}")
            return self._extract_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {str(e)}")
            raise
    
    def _extract_with_llama_parse(self, file_path: str) -> List[Document]:
        """Extract using LlamaParse with enhanced metadata."""
        try:
            # Get raw results from LlamaParse
            raw_results = self.llama_parser.load_data(file_path)
            
            # Convert to Langchain Documents with proper attributes
            documents = []
            for result in raw_results:
                # LlamaParse returns dict-like objects, we need to convert them to Documents
                content = result.get('content', '') if isinstance(result, dict) else str(result)
                metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
                
                # Create Document with required page_content
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': file_path,
                        **metadata
                    }
                )
                
                # Add structure recognition
                structured_content = self.doc_structure.extract_content(content)
                doc.metadata['structured_content'] = structured_content
                doc.metadata['extraction_method'] = 'llama_parse'
                
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in LlamaParse extraction for {file_path}: {str(e)}")
            # Fall back to pdfplumber if LlamaParse fails
            return self._extract_with_pdfplumber(file_path)
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Document]:
        """Extract using pdfplumber with enhanced metadata."""
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        try:
                            # Add structure recognition with error handling
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
                        except Exception as e:
                            logger.warning(f"Error extracting structure from page {page_num} of {file_path}: {str(e)}")
                            # Still create document even if structure extraction fails
                            doc = Document(
                                page_content=text,
                                metadata={
                                    'source': file_path,
                                    'page': page_num,
                                    'extraction_method': 'pdfplumber'
                                }
                            )
                            documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            # Create a basic document with the error information
            return [Document(
                page_content=f"Error processing document: {str(e)}",
                metadata={
                    'source': file_path,
                    'error': str(e),
                    'extraction_method': 'pdfplumber_error'
                }
            )]
    
class ProgressEmbeddings(OllamaEmbeddings):
    """Adds progress tracking to embedding generation."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with progress bar."""
        logger.info(f"Creating embeddings for {len(texts)} text chunks...")
        
        results = []        
        with tqdm(total=len(texts), desc="Creating embeddings") as pbar:
            for text in texts:
                # Call the parent class's embed_query method directly
                embedding = super().embed_query(text)
                results.append(embedding)
                pbar.update(1)
                time.sleep(0.1)  # Rate limiting
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # Call parent's embed_query directly
        return super().embed_query(text)

class RAGPrototype:
    """Enhanced RAG system with structure awareness."""
    
    def __init__(self, pdf_directory: str, llama_parse_api_key: Optional[str] = None, custom_patterns: Optional[Dict] = None):
        """Initialize RAG system."""
        try:
            # Store the API key and directory
            self.llama_parse_api_key = llama_parse_api_key
            self.pdf_directory = pdf_directory
            
            # Initialize embedding model
            self.embeddings = OllamaEmbeddings(
                model="llama3",
                base_url="http://localhost:11434"
            )
            
            # Test embedding functionality
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embedding model initialized successfully. Embedding dimension: {len(test_embedding)}")
            
            # Initialize LLM
            self.llm = OllamaLLM(
                model="llama3",
                temperature=0.1,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            # Initialize document structure with all patterns
            all_patterns = {}
            
            # Start with default patterns from DocumentStructure
            all_patterns.update(DocumentStructure.DEFAULT_PATTERNS)
            
            # Add custom patterns defined in DocumentStructure
            all_patterns.update(DocumentStructure.CUSTOM_PATTERNS)
            
            # Add any additional patterns passed during initialization
            if custom_patterns:
                all_patterns.update(custom_patterns)
            
            # Log all active patterns
            logger.info("Initializing with patterns:")
            for pattern_name in all_patterns.keys():
                logger.info(f"  - {pattern_name}")
            
            # Initialize document structure and classifier with combined patterns
            self.doc_structure = DocumentStructure(all_patterns)
            self.doc_classifier = DocumentClassifier()
            
            # Initialize PDF extractor
            self.pdf_extractor = PDFExtractor(
                llama_parse_api_key=llama_parse_api_key,
                custom_patterns=all_patterns
            )
            
            # Initialize empty documents list
            self.documents = []
            self.vectorstore = None
            
            # Initialize structured_content dictionary
            self.structured_content = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
    
    def load_documents(self) -> None:
        """Load and process documents from the PDF directory."""
        try:
            logger.info("Loading documents...")
            self.documents = []
            self.structured_content = {}  # Reset structured content
            
            for filename in os.listdir(self.pdf_directory):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(self.pdf_directory, filename)
                    
                    try:
                        logger.info(f"Processing {filename}...")
                        docs = self.pdf_extractor.extract_from_pdf(file_path)
                        
                        # Prioritize and extract important sections
                        for doc in docs:
                            important_sections = {}
                            for section in ['action_items', 'motions', 'attendees']:
                                content = self.doc_structure.extract_content(doc.page_content, pattern=section)
                                if content:
                                    important_sections[section] = content
                            
                            # Store extracted sections in metadata
                            doc.metadata['important_sections'] = important_sections
                            
                            self.documents.append(doc)
                        
                        logger.info(f"Successfully processed {filename} - {len(docs)} documents extracted")

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        continue

            if not self.documents:
                raise ValueError("No documents were successfully processed")
            
            # Create vector store for efficient retrieval
            self._create_vectorstore()

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def _process_with_pdfplumber(self, file_path: str) -> List[Document]:
        """Process PDF with PDFPlumber as fallback."""
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            'source': file_path,
                            'page': page_num + 1,
                            'structured_content': self.doc_structure.extract_content(text),  # Use extract_content
                            'extraction_method': 'pdfplumber'
                        }
                    ))
        return docs
    
    @staticmethod
    def _validate_metadata_type(value) -> bool:
        """Validate if a metadata value is of an acceptable type."""
        return isinstance(value, (str, int, float, bool))

    def _create_vectorstore(self):
        """Create vector store from processed documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1026,
            chunk_overlap=256
        )
    
        # Split documents
        split_docs = text_splitter.split_documents(self.documents)
    
        # Filter complex metadata
        for i, doc in enumerate(split_docs):
            logger.debug(f"Processing document {i + 1}/{len(split_docs)}")
            logger.debug(f"Original metadata: {doc.metadata}")

            # Convert structured content to string representation
            if 'structured_content' in doc.metadata:
                doc.metadata['structured_content_summary'] = str(doc.metadata['structured_content'])
                del doc.metadata['structured_content']
    
            # First pass: filter complex metadata
            doc.metadata = filter_complex_metadata(doc.metadata)
    
            # Second pass: validate all metadata values
            invalid_fields = {
                key: type(value).__name__
                for key, value in doc.metadata.items()
                if not self._validate_metadata_type(value)
            }
    
            if invalid_fields:
                logger.warning(f"Found invalid metadata types: {invalid_fields}")
                # Convert any remaining invalid types to strings
                for key, _ in invalid_fields.items():
                    doc.metadata[key] = str(doc.metadata[key])
    
            # Final validation
            assert all(self._validate_metadata_type(value) for value in doc.metadata.values()), \
                f"Invalid metadata types remain: {doc.metadata}"
    
            logger.debug(f"Filtered metadata: {doc.metadata}")
    
        logger.info(f"Processed {len(split_docs)} documents")
    
        # Create vectorstore with filtered metadata
        try:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings
            )
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            # Log the problematic metadata
            for i, doc in enumerate(split_docs):
                logger.error(f"Document {i} metadata: {doc.metadata}")
            raise

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
            
            # Use invoke instead of __call__ and use 'query' instead of 'question'
            response = qa_chain.invoke({"query": question, "context": context})
            
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
            'time': ['meeting_times, call_to_order'],
            'when': ['meeting_times, close_of_meeting'],
            'action': ['action_items'],
            'motion': ['motions'],
            'attend': ['attendees'],
            'present': ['attendees'],
            'regrets': ['regrets'],
            'business': ['new_business'],
            'management': ['management_report'],
            'financial': ['financial_report', 'statements', 'arrears_report', 'review_of_unaudited_financial_statements', 'review_and_approval_of_meeting_minutes', 'variance_report']
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
        Answer the question based only on the provided context. Use a conversational yet professional tone. Do not reference the document directly in your answer. If the user requests, offer to provide the document name (not the chunk ID). If you cannot answer the question based on the context, say "I cannot answer this based on the provided documents."

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

    def query_with_important_sections(self, question: str) -> str:
        """Query the LLM using prioritized sections."""
        context_parts = []
        
        for doc in self.documents:
            important_sections = doc.metadata.get('important_sections', {})
            for section, content in important_sections.items():
                context_parts.append(f"{section.capitalize()}:\n{content}")
        
        context = "\n\n".join(context_parts)
        response = self.llm.query(context, question)
        
        return response