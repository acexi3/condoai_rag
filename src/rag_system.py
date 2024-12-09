"""
RAG (Retrieval Augmented Generation) System for Condo Document Analysis
"""

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from llama_parse import LlamaParse
import pdfplumber
from langchain.schema import Document
from langchain.prompts import PromptTemplate

import os
from typing import List, Optional, Dict
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Handles PDF extraction using PDFPlumber"""
    
    @staticmethod
    def extract_pdf_content(pdf_path: str) -> Dict:
        """Extract text, tables, and images from PDF."""
        content = {
            'text': [],
            'tables': [],
            'images': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        content['text'].append({
                            'page': page_num,
                            'content': text
                        })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        content['tables'].extend([{
                            'page': page_num,
                            'content': table
                        } for table in tables])
                    
                    # Extract images (if available)
                    try:
                        images = page.images
                        if images:
                            content['images'].extend([{
                                'page': page_num,
                                'type': img.get('type', 'unknown')
                            } for img in images])
                    except AttributeError:
                        logger.warning(f"No image extraction support for page {page_num}")
                        
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {str(e)}")
            raise
            
        return content

class ProgressEmbeddings(OllamaEmbeddings):
    """
    Wrapper around OllamaEmbeddings to add progress tracking.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of text chunks with progress tracking."""
        logger.info(f"Creating embeddings for {len(texts)} text chunks...")
        results = []
        with tqdm(total=len(texts), desc="Creating embeddings") as pbar:
            for text in texts:
                result = super().embed_documents([text])[0]
                results.append(result)
                pbar.update(1)
                time.sleep(0.1)  # Prevent overwhelming the API
        return results

class RAGPrototype:
    def __init__(
            self, 
            pdf_directory: str, 
            llama_parse_api_key: str
        ):
        self.pdf_directory = pdf_directory
        self.pdf_extractor = PDFExtractor()
        
        try:
            self.llm = OllamaLLM(
                model="llama3",
                callbacks=[StreamingStdOutCallbackHandler()],
                temperature=0.5
            )
            self.embeddings = ProgressEmbeddings(model="llama3")
            self.parser = LlamaParse(
                api_key=llama_parse_api_key,
                result_type="markdown"
            )
            logger.info("Successfully initialized components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
            
        self.vector_store = None
        
    def _process_extracted_content(
            self, 
            content: Dict
        ) -> List[Document]:
        """Process extracted content into documents."""
        documents = []
        
        # Process text
        for text_item in content['text']:
            documents.append(Document(
                page_content=text_item['content'],
                metadata={'page': text_item['page'], 'type': 'text'}
            ))
        
        # Process tables
        for table_item in content['tables']:
            # Convert table to string representation
            table_str = '\n'.join(['\t'.join(map(str, row)) for row in table_item['content']])
            documents.append(Document(
                page_content=table_str,
                metadata={'page': table_item['page'], 'type': 'table'}
            ))
        
        return documents
        
    def load_documents(
            self, 
            persist: bool = True
        ) -> None:
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"Directory {self.pdf_directory} does not exist. Creating it...")
            os.makedirs(self.pdf_directory)
            return
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return
            
        for file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                pdf_path = os.path.join(self.pdf_directory, file)
                
                # Extract content using PDFPlumber
                content = self.pdf_extractor.extract_pdf_content(pdf_path)
                
                # Process extracted content
                doc_chunks = self._process_extracted_content(content)
                documents.extend(doc_chunks)
                
                logger.info(f"Loaded and processed: {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        if not documents:
            logger.warning("No documents were successfully loaded")
            return
            
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        # Create vector store
        persist_directory = "./chroma_db"
        
        try:
            logger.info("Creating vector store...")
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=persist_directory if persist else None
            )
            
            if persist:
                logger.info("Persisting vector store...")
                self.vector_store.persist()
                logger.info(f"Vector store persisted to {persist_directory}")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def _determine_k(self, question: str) -> int:
        """
        Determine number of chunks to retrieve based on question complexity.
        
        Args:
            question: The user's question
            
        Returns:
            int: Recommended number of chunks to retrieve
        """
        # Base k value
        k = 4
        
        # Keywords indicating need for more context
        cross_doc_keywords = [
            'across', 'all documents', 'compare', 'differences',
            'multiple', 'various', 'each', 'different'
        ]
        
        financial_keywords = [
            'budget', 'financial', 'costs', 'expenses',
            'spending', 'funds', 'money', 'payments'
        ]
        
        timeline_keywords = [
            'when', 'history', 'timeline', 'schedule',
            'dates', 'planned', 'future', 'past'
        ]
        
        # Increase k based on question complexity
        if any(keyword in question.lower() for keyword in cross_doc_keywords):
            k += 4  # Need more chunks for cross-document analysis
            
        if any(keyword in question.lower() for keyword in financial_keywords):
            k += 2  # Financial info might be spread across documents
            
        if any(keyword in question.lower() for keyword in timeline_keywords):
            k += 2  # Temporal questions might need more context
            
        # Cap k at reasonable maximum
        return min(k, 16)

    def query(self, 
              question: str, 
              k: int = 4, 
              include_sources: bool = False) -> dict:
        """
        Query the RAG system with dynamic chunk retrieval.
        
        Args:
            question: The user's question
            k: Number of relevant chunks to retrieve
            include_sources: Whether to return source documents
            
        Returns:
            Dictionary containing answer and optionally sources
        """
        if not self.vector_store:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Determine k dynamically based on question complexity
            k = self._determine_k(question)
            logger.info(f"Using k={k} chunks based on question complexity")
            
            # Enhanced prompt template
            prompt_template =  """
            Please analyze the provided documents carefully and answer the question below.
            
            Important instructions:
            1. Clearly distinguish between:
            - Definitive statements (IS, MUST, SHALL)
            - Conditional possibilities (MAY, MIGHT, COULD)
            - Recommendations or suggestions (SHOULD)
            
            2. For each point in your answer:
            - Cite the specific document and section
            - Quote relevant text when appropriate
            - Indicate if something is a requirement or a possibility
            
            3. If information appears in multiple documents:
            - Note any differences or contradictions
            - Cite all relevant sources
            
            Question: {question}
            
            Context: {context}
            
            Please provide a structured answer that clearly separates definitive requirements from possible or conditional statements.
            """

            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Combines all chunks into one context
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                return_source_documents=include_sources,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )}
            )
            
            # Get response
            response = qa_chain.invoke({"query": question})
            
            return {
                "answer": response["result"],
                "sources": response["source_documents"] if include_sources else None
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise