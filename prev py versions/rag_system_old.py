"""
RAG (Retrieval Augmented Generation) System for Condo Document Analysis
--------------------------------------------------------------------

This module implements a RAG system that:
1. Loads and processes PDF documents using LlamaParse
2. Creates embeddings (vector representations) of document chunks
3. Stores these embeddings in a vector database (Chroma)
4. Retrieves relevant context when answering questions
5. Generates responses using the Llama 3 language model

The RAG approach enhances the LLM's responses by:
- Grounding answers in your specific documents
- Reducing hallucination (making up facts)
- Providing source attribution for answers
"""

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from llama_parse import LlamaParse
from llama_parse.plugins import PDFPlumberTextPlugin

import os
from typing import List, Optional
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressEmbeddings(OllamaEmbeddings):
    """
    Wrapper around OllamaEmbeddings to add progress tracking.
    
    This class extends OllamaEmbeddings to show progress during the embedding
    creation process, which can be time-consuming for large documents.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks with progress tracking.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            List of embedding vectors (high-dimensional float arrays)
        """
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
    """
    Main RAG system implementation.
    
    This class handles:
    1. Document loading and parsing with LlamaParse
    2. Text chunking for optimal context windows
    3. Embedding creation and storage
    4. Question answering using retrieved context
    """
    
    def __init__(self, pdf_directory: str, llama_parse_api_key: str):
        """
        Initialize the RAG system.
        
        Args:
            pdf_directory: Path to directory containing PDF documents
        """
        self.pdf_directory = pdf_directory
        
        # Initialize components
        try:
            self.llm = OllamaLLM(
                model="llama3",
                callbacks=[StreamingStdOutCallbackHandler()],  # Updated from callback_manager
                temperature=0.5
            )
            # Initialize embeddings model
            self.embeddings = ProgressEmbeddings(model="llama3")

            # Initialize LlamaParse for better PDF handling
            self.parser = LlamaParse(
                api_key=llama_parse_api_key,
                plugins=[PDFPlumberTextPlugin()],
                result_type="markdown"
            )

            logger.info("Successfully initialized Ollama with Llama 3")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise
            
        self.vector_store = None
        
    def load_documents(self, persist: bool = True) -> None:
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
                parsed_doc = self.parser.load_data(pdf_path)
                documents.extend(parsed_doc)
                logger.info(f"Loaded and parsed: {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        if not documents:
            logger.warning("No documents were successfully loaded")
            return
            
        # Split documents into chunks
        # This is crucial for:
        # 1. Staying within context windows
        # 2. Creating meaningful semantic units
        # 3. Enabling precise retrieval
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,  # Larger chunks for better context
            chunk_overlap=200,  # Overlap to maintain context across chunks
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        # Create and store embeddings
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

    def query(self, 
              question: str, 
              k: int = 4, 
              include_sources: bool = False) -> dict:
        """
        Query the RAG system.
        
        Process:
        1. Convert question to embedding
        2. Retrieve relevant document chunks
        3. Combine chunks with question for context
        4. Generate answer using LLM
        
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
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Combines all chunks into one context
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                return_source_documents=include_sources
            )
            
            # Get response
            response = qa_chain({"query": question})
            
            return {
                "answer": response["result"],
                "sources": [doc.page_content for doc in response["source_documents"]] if include_sources else None
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise