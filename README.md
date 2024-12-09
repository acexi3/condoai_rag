# CondoAI RAG System

A Retrieval Augmented Generation (RAG) system designed for analyzing condominium documents using LLM technology.

## Overview

This system processes and analyzes condominium-related documents (such as bylaws, minutes, management reports, and owner's guides) using advanced natural language processing. It can answer questions about condo governance, maintenance responsibilities, financial matters, and more while maintaining context and accuracy.

## Features

- **PDF Processing**: Extracts text, tables, and images from PDF documents
- **Dynamic Context Retrieval**: Adjusts the amount of context based on question complexity
- **Source Attribution**: Provides references to source documents and specific sections
- **Distinction Handling**: Clearly separates definitive requirements from conditional possibilities
- **Cross-Document Analysis**: Compares information across multiple documents

## Prerequisites

- Python 3.8+
- Ollama (for LLM)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/condoai_rag.git
cd condoai_rag

2. Install required packages:

bash
pip install -r requirements.txt

3. Set up environment variables:

cp .env.example .env
Edit .env with your configuration

## Usage

1. Place PDF documents in the `pdfs` directory

2. Run the system: python src/test_rag.py

3. Query the system:
    python
    from rag_system import RAGPrototype
    rag = RAGPrototype(pdf_directory="pdfs", llama_parse_api_key="your_api_key")
    rag.load_documents()
    response = rag.query("What are the maintenance responsibilities?")
    print(response["answer"])

## Project Structure

condoai_rag/
├── src/
│ ├── rag_system.py # Main RAG implementation
│ └── test_rag.py # Test suite
├── pdfs/ # Directory for PDF documents
├── chroma_db/ # Vector store directory
├── requirements.txt # Python dependencies
└── README.md # This file


## Key Components

- **PDFExtractor**: Handles PDF document processing
- **ProgressEmbeddings**: Manages document embedding with progress tracking
- **RAGPrototype**: Main class implementing the RAG system
- **Dynamic Context**: Adjusts retrieval based on question complexity

## Configuration

The system can be configured through environment variables:
- `LLAMA_PARSE_API_KEY`: Your API key for LlamaParse
- Additional configuration options in `.env`

## Testing

Run the test suite: python src/test_rag.py

The test suite includes various question types to evaluate:
- Document comprehension
- Cross-document analysis
- Distinction between requirements and possibilities
- Financial information processing
- Meeting minutes analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Copyright © 2024 Douglas Baker. All rights reserved.

This is proprietary software. No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the owner.

Unauthorized copying, modification, distribution, or use of this software is strictly prohibited.

## Acknowledgments

- LangChain for the core RAG functionality
- Ollama for LLM support
- ChromaDB for vector storage
- PDFPlumber for PDF processing

## Contact

Douglas Baker - [@dwbaker1971](https://x.com/dwbaker1971)
Project Link: [https://github.com/acexi3/condoai_rag](https://github.com/acexi3/condoai_rag)