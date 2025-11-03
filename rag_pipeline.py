"""
Multi-Modal RAG Pipeline

A comprehensive Retrieval-Augmented Generation pipeline that processes multi-modal
documents (PDF, PPTX, EML, Excel) and provides intelligent question-answering
capabilities using Google Vertex AI.

Features:
- Multi-format document processing with AI-powered content extraction
- Separate vector storage for text, tables, and visual content
- Intelligent query analysis and contextual retrieval
- Parallel processing for performance optimization
- Comprehensive caching and performance monitoring

"""

from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from datetime import datetime

from langchain_google_vertexai import ChatVertexAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from document_processor import DocumentProcessor
from vector_store import MultiModalVectorStore
from config import Config


class MultiModalRAGPipeline:
    """
    Advanced RAG pipeline for multi-modal document processing and question answering.
    
    Integrates document processing, vector storage, and AI-powered retrieval
    to provide comprehensive question-answering capabilities across different
    content types (text, tables, visuals).
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with all required components."""
        print("Initializing Multi-Modal RAG Pipeline...")
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.vector_store = MultiModalVectorStore()
        self._initialize_ai_generator()
        
        print("✓ RAG Pipeline initialized successfully")
    
    # =============================================================================
    # SYSTEM INITIALIZATION
    # =============================================================================
    
    def _initialize_ai_generator(self) -> None:
        """
        Initialize the AI text generation system with Google Vertex AI.
        
        Uses Gemini 2.5-pro for high-quality answer generation.
        """
        try:
            # Configure Google Cloud credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            # Initialize text generation model
            self.answer_generator = ChatVertexAI(
                model_name="gemini-2.5-pro",
                temperature=0.1,  # Low temperature for consistent, factual responses
                project=Config.GOOGLE_PROJECT_ID,
                location=Config.GOOGLE_LOCATION
            )
            
            print("✓ AI answer generation system initialized")
            
        except Exception as e:
            raise Exception(f"Failed to initialize AI generator: {e}")
    
    # =============================================================================
    # DOCUMENT INGESTION OPERATIONS
    # =============================================================================
    
    def ingest_single_document(self, file_path: str, force_reprocess: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ingest a single document into the RAG pipeline with intelligent caching.
        
        Args:
            file_path: Path to the document file
            force_reprocess: Whether to bypass cache and reprocess the document
            
        Returns:
            Dictionary containing processed content organized by type
        """
        print(f"Ingesting document: {os.path.basename(file_path)}")
        
        # Reset duplicate detection for new document processing
        self.document_processor.reset_duplicate_detection()
        
        # Check for cached processing results
        if not force_reprocess and self.document_processor.has_valid_cache(file_path):
            print("Loading from cached extraction...")
            processed_content = self.document_processor.load_from_saved_extraction(file_path)
        else:
            # Process document with optimal strategy
            if Config.ENABLE_MULTIPROCESSING:
                processed_content = self.document_processor.process_document_with_parallel_processing(
                    file_path, force_reprocess
                )
            else:
                processed_content = self.document_processor.process_document(
                    file_path, force_reprocess
                )
        
        # Store processed content in vector collections
        self.vector_store.store_processed_documents(processed_content)
        
        print("✓ Document successfully ingested into RAG pipeline")
        return processed_content
    
    def ingest_folder_of_documents(self, folder_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process and ingest all supported documents in a folder.
        
        Args:
            folder_path: Path to folder containing documents
            force_reprocess: Whether to bypass caching
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Find all supported document files
        supported_extensions = ['.pdf', '.pptx', '.eml', '.xls', '.xlsx']
        document_files = [
            os.path.join(folder_path, filename) 
            for filename in os.listdir(folder_path) 
            if any(filename.lower().endswith(ext) for ext in supported_extensions)
        ]
        
        if not document_files:
            print("No supported document files found in folder")
            return {"processed_files": [], "total_documents": 0}
        
        print(f"Found {len(document_files)} supported files to process")
        
        # Process each document
        processing_results = {}
        total_documents_stored = 0
        
        for file_path in document_files:
            try:
                processed_content = self.ingest_single_document(file_path, force_reprocess)
                if processed_content:
                    processing_results[file_path] = processed_content
                    
                    # Count total documents
                    document_count = (
                        len(processed_content.get('text', [])) + 
                        len(processed_content.get('tables', [])) + 
                        len(processed_content.get('visuals', []))
                    )
                    total_documents_stored += document_count
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                processing_results[file_path] = None
        
        successful_files = len([r for r in processing_results.values() if r is not None])
        
        print(f"✓ Successfully processed {successful_files}/{len(document_files)} files")
        print(f"✓ Total documents stored: {total_documents_stored}")
        
        return {
            "processed_files": list(processing_results.keys()),
            "successful_files": successful_files,
            "total_files": len(document_files),
            "total_documents": total_documents_stored,
            "results": processing_results
        }
    
    def load_from_cached_extractions(self, extraction_directory: str = "document_extractions/content") -> Tuple[int, int]:
        """
        Load documents from previously saved extraction files.
        
        Args:
            extraction_directory: Directory containing cached extraction files
            
        Returns:
            Tuple of (successful_loads, total_files)
        """
        print("Loading documents from cached extractions...")
        return self.vector_store.bulk_load_from_extraction_directory(extraction_directory)
    
    
    # =============================================================================
    # QUESTION ANSWERING OPERATIONS
    # =============================================================================
    
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate a comprehensive answer to a user question using RAG.
        
        Uses intelligent retrieval to find the most relevant content across
        all content types, then generates a well-structured answer.
        
        Args:
            question: User's question to answer
            
        Returns:
            Dictionary containing answer, sources, and supporting materials
        """
        print(f"Processing question: '{question}'")
        
        # Retrieve contextually relevant documents using AI-driven strategy
        relevant_documents = self.vector_store.get_contextually_relevant_documents(question)
        
        if not relevant_documents:
            return {
                "answer": "I couldn't find relevant information in the ingested documents to answer your question. Please ensure the documents containing the relevant information have been ingested into the system.",
                "sources": [],
                "context_used": "",
                "supporting_images": [],
                "supporting_tables": [],
                "confidence": "low"
            }
        
        # Format retrieved content into structured context
        formatted_context = self._format_context_for_generation(relevant_documents)
        
        # Generate comprehensive answer using AI
        answer_content = self._generate_ai_response(question, formatted_context)
        
        # Extract supporting materials
        source_references = self._extract_source_references(relevant_documents)
        supporting_images = self._extract_image_references(relevant_documents)
        supporting_tables = self._extract_table_content(relevant_documents)
        
        print(f"✓ Generated answer with {len(source_references)} sources")
        
        return {
            "answer": answer_content,
            "sources": source_references,
            "context_used": formatted_context,
            "supporting_images": supporting_images,
            "supporting_tables": supporting_tables,
            "confidence": "high" if len(relevant_documents) >= 3 else "medium"
        }
    
    def _format_context_for_generation(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into well-structured context for AI generation.
        
        Args:
            documents: List of relevant documents from vector search
            
        Returns:
            Formatted context string organized by content type
        """
        context_sections = []
        
        # Organize documents by content type
        text_documents = [d for d in documents if d.metadata.get("content_type") == "text"]
        table_documents = [d for d in documents if d.metadata.get("content_type") == "table"]
        visual_documents = [d for d in documents if d.metadata.get("content_type") == "visual"]
        
        # Format text content section
        if text_documents:
            context_sections.append("=== TEXTUAL INFORMATION ===")
            for doc in text_documents:
                page_ref = doc.metadata.get('page_number', 'Unknown')
                source_ref = os.path.basename(doc.metadata.get('source_file', 'Unknown'))
                context_sections.append(f"From {source_ref}, Page {page_ref}:")
                context_sections.append(doc.page_content)
                context_sections.append("")
        
        # Format table content section
        if table_documents:
            context_sections.append("=== STRUCTURED DATA & TABLES ===")
            for doc in table_documents:
                page_ref = doc.metadata.get('page_number', 'Unknown')
                source_ref = os.path.basename(doc.metadata.get('source_file', 'Unknown'))
                context_sections.append(f"Table from {source_ref}, Page {page_ref}:")
                context_sections.append(doc.page_content)
                context_sections.append("")
        
        # Format visual content section
        if visual_documents:
            context_sections.append("=== VISUAL ELEMENTS & CHARTS ===")
            for doc in visual_documents:
                page_ref = doc.metadata.get('page_number', 'Unknown')
                source_ref = os.path.basename(doc.metadata.get('source_file', 'Unknown'))
                context_sections.append(f"Visual description from {source_ref}, Page {page_ref}:")
                context_sections.append(doc.page_content)
                context_sections.append("")
        
        return "\n".join(context_sections)
    
    def _generate_ai_response(self, question: str, context: str) -> str:
        """
        Generate AI response using the formatted context.
        
        Args:
            question: User's original question
            context: Formatted context from retrieved documents
            
        Returns:
            Generated answer content
        """
        generation_prompt = f"""
        You are an expert analyst with access to multi-modal document content. Based on the provided context, answer the user's question comprehensively and accurately.
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        USER QUESTION: {question}
        
        INSTRUCTIONS:
        1. Provide a complete, well-structured answer based on the context
        2. Use information from text, tables, and visual descriptions as appropriate
        3. When referencing data from tables, present it clearly and cite the source
        4. When mentioning visual elements, describe them based on the visual descriptions
        5. Always cite which document and page your information comes from
        6. If the context doesn't fully answer the question, clearly state what's missing
        7. Organize your response with clear paragraphs and bullet points where helpful
        8. Be precise and factual - don't add information not present in the context
        
        ANSWER:
        """
        
        try:
            ai_response = self.answer_generator.invoke([HumanMessage(content=generation_prompt)])
            return ai_response.content.strip()
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "I encountered an error while generating the response. Please try asking your question again."
    
    def _extract_source_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract unique source references from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of unique source reference dictionaries
        """
        source_references = []
        seen_sources = set()
        
        for doc in documents:
            source_file = doc.metadata.get("source_file", "Unknown")
            page_number = doc.metadata.get("page_number", "Unknown")
            content_type = doc.metadata.get("content_type", "Unknown")
            
            source_key = f"{source_file}_{page_number}_{content_type}"
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                source_references.append({
                    "source_file": os.path.basename(source_file),
                    "page_number": page_number,
                    "content_type": content_type,
                    "full_path": source_file
                })
        
        return source_references
    
    def _extract_image_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract unique image references from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of unique image reference dictionaries
        """
        image_references = []
        seen_images = set()
        
        for doc in documents:
            image_path = doc.metadata.get("image_reference", "")
            if image_path and image_path not in seen_images:
                seen_images.add(image_path)
                image_references.append({
                    "image_path": image_path,
                    "page_number": doc.metadata.get("page_number", "Unknown"),
                    "source_file": os.path.basename(doc.metadata.get("source_file", "Unknown")),
                    "content_type": doc.metadata.get("content_type", "Unknown")
                })
        
        return image_references
    
    def _extract_table_content(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract table content and metadata from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of table content dictionaries
        """
        table_content = []
        
        for doc in documents:
            if doc.metadata.get("content_type") == "table":
                table_content.append({
                    "table_data": doc.page_content,
                    "page_number": doc.metadata.get("page_number", "Unknown"),
                    "source_file": os.path.basename(doc.metadata.get("source_file", "Unknown")),
                    "image_reference": doc.metadata.get("image_reference", "")
                })
        
        return table_content
        """Ingest a document with page-by-page logging and display"""
        from document_processor import DocumentProcessor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"Processing document: {file_path}")
        
        # Get document processor
        processor = self.document_processor
        
        # Convert document to images
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            images = processor.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = processor.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        if Config.ENABLE_MULTIPROCESSING and len(images) >= 2:
            # Parallel processing with logging
            print(f"Processing {len(images)} pages with {Config.MAX_WORKERS} workers...")
            
            # Prepare page data
            page_data_list = [
                {
                    'image': image,
                    'page_num': page_num + 1,
                    'file_path': file_path
                }
                for page_num, image in enumerate(images)
            ]
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(processor.process_page, page_data): page_data['page_num']
                    for page_data in page_data_list
                }
                
                results = {}
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        results[result['page_num']] = result
                        print(f"✓ Completed page {result['page_num']}/{len(images)}")
                    except Exception as e:
                        print(f"✗ Error processing page {page_num}: {e}")
                        results[page_num] = {
                            'page_num': page_num,
                            'text': None,
                            'table': None,
                            'visual': None
                        }
            
            # Display and log results in page order
            for page_num in sorted(results.keys()):
                result = results[page_num]
                
                text_content = result['text']['content'] if result['text'] else ""
                table_content = result['table']['content'] if result['table'] else ""
                visual_content = result['visual']['content'] if result['visual'] else ""
                
                # Display and log
                display_callback(page_num, text_content, table_content, visual_content, log_file)
                
                # Collect processed content
                if result['text']:
                    processed_content["text"].append(result['text'])
                if result['table']:
                    processed_content["tables"].append(result['table'])
                if result['visual']:
                    processed_content["visuals"].append(result['visual'])
        else:
            # Sequential processing (original logic)
            for page_num, image in enumerate(images):
                print(f"\nProcessing page {page_num + 1}/{len(images)}...")
                
                # Extract text content
                text_content = processor.extract_content_from_image(image, "text")
                if text_content.strip() and "no text content found" not in text_content.lower():
                    processed_content["text"].append({
                        "content": text_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "text"
                    })
                
                # Extract table content
                table_content = processor.extract_content_from_image(image, "table")
                if table_content.strip() and "no tables found" not in table_content.lower():
                    processed_content["tables"].append({
                        "content": table_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "table"
                    })
                
                # Extract visual content
                visual_content = processor.extract_content_from_image(image, "visual")
                if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                    processed_content["visuals"].append({
                        "content": visual_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "visual"
                    })
                
                # Display and log the extracted content for this page
                display_callback(page_num + 1, text_content, table_content, visual_content, log_file)
        
        # Add all processed content to vector stores
        self.vector_store.add_documents(processed_content)
        
        print("\nDocument successfully ingested!")
        
        # Return summary for logging
        summary = f"\n{'='*80}\n"
        summary += f"DOCUMENT PROCESSING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Pages/Slides: {len(images)}\n"
        summary += f"Text Chunks: {len(processed_content['text'])}\n"
        summary += f"Table Entries: {len(processed_content['tables'])}\n"
        summary += f"Visual Descriptions: {len(processed_content['visuals'])}\n"
        summary += f"Processing Mode: {'Parallel' if Config.ENABLE_MULTIPROCESSING else 'Sequential'}\n"
        summary += f"{'='*80}\n"
        
        log_file.write(summary)
        return summary
    
    def ingest_document_with_parallel_logging(self, file_path: str, log_file, display_callback):
        """Ingest a document with parallel page processing and real-time progress"""
        from document_processor import DocumentProcessor
        
        print(f"Processing document: {file_path}")
        
        # Get document processor
        processor = self.document_processor
        
        # Convert document to images
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            images = processor.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = processor.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        total_pages = len(images)
        print(f"Total pages to process: {total_pages}")
        print(f"Using parallel processing with up to 3 workers...")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        # Create a lock for thread-safe operations
        lock = Lock()
        
        def process_page(page_num, image, pbar):
            """Process a single page"""
            try:
                # Extract all content types
                text_content = processor.extract_content_from_image(image, "text")
                table_content = processor.extract_content_from_image(image, "table")
                visual_content = processor.extract_content_from_image(image, "visual")
                
                # Prepare results
                page_results = {
                    "text": None,
                    "table": None,
                    "visual": None,
                    "page_num": page_num
                }
                
                if text_content.strip() and "no text content found" not in text_content.lower():
                    page_results["text"] = {
                        "content": text_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "text"
                    }
                
                if table_content.strip() and "no tables found" not in table_content.lower():
                    page_results["table"] = {
                        "content": table_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "table"
                    }
                
                if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                    page_results["visual"] = {
                        "content": visual_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "visual"
                    }
                
                # Display and log with lock
                with lock:
                    display_callback(page_num + 1, text_content, table_content, visual_content, log_file, lock)
                
                # Update progress bar
                pbar.update(1)
                
                return page_results
            except Exception as e:
                pbar.update(1)
                raise e
        
        # Process pages in parallel with tqdm progress bar
        page_data = []
        
        # Create progress bar
        with tqdm(total=total_pages, desc="📄 Extracting Pages", unit="page", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                 ncols=100) as pbar:
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all pages for processing
                futures = [
                    executor.submit(process_page, idx, img, pbar)
                    for idx, img in enumerate(images)
                ]
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        page_data.append((result["page_num"], result))
                    except Exception as e:
                        tqdm.write(f"✗ Error processing page: {e}")
        
        print()  # New line after progress bar
        
        # Sort by page number and add to processed_content
        page_data.sort(key=lambda x: x[0])
        for page_num, result in page_data:
            if result["text"]:
                processed_content["text"].append(result["text"])
            if result["table"]:
                processed_content["tables"].append(result["table"])
            if result["visual"]:
                processed_content["visuals"].append(result["visual"])
        
        # Add all processed content to vector stores
        print("Adding extracted content to vector stores...")
        self.vector_store.add_documents(processed_content)
        
        print("✓ Document successfully ingested!")
        
        # Return summary for logging
        summary = f"\n{'='*80}\n"
        summary += f"DOCUMENT PROCESSING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Pages/Slides: {total_pages}\n"
        summary += f"Text Chunks: {len(processed_content['text'])}\n"
        summary += f"Table Entries: {len(processed_content['tables'])}\n"
        summary += f"Visual Descriptions: {len(processed_content['visuals'])}\n"
        summary += f"Processing Mode: Parallel (3 workers)\n"
        summary += f"{'='*80}\n"
        
        log_file.write(summary)
        return summary
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context for the LLM"""
        context_parts = []
        
        # Group documents by type
        text_docs = [d for d in documents if d.metadata.get("type") == "text"]
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        visual_docs = [d for d in documents if d.metadata.get("type") == "visual"]
        
        # Format text content
        if text_docs:
            context_parts.append("=== TEXT CONTENT ===")
            for doc in text_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Format table content
        if table_docs:
            context_parts.append("=== TABLE DATA ===")
            for doc in table_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Format visual content
        if visual_docs:
            context_parts.append("=== VISUAL INFORMATION ===")
            for doc in visual_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_relevant_images(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract unique image paths from retrieved documents"""
        image_references = []
        seen_images = set()
        
        for doc in documents:
            image_path = doc.metadata.get("image_path", "")
            if image_path and image_path not in seen_images:
                seen_images.add(image_path)
                image_references.append({
                    "image_path": image_path,
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown")
                })
        
        return image_references
    
    def extract_tables_from_context(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract table content and metadata from documents"""
        tables = []
        
        for doc in documents:
            if doc.metadata.get("type") == "table":
                tables.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "image_path": doc.metadata.get("image_path", "")
                })
        
        return tables
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.get_contextual_documents(question)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "context_used": "",
                "relevant_images": [],
                "relevant_tables": []
            }
        
        # Format context
        context = self.format_context(relevant_docs)
        
        # Create prompt with context
        prompt = f"""
        Based on the following context from documents, please answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Use information from text, tables, and visual descriptions as needed
        - If the answer involves data from tables, present it clearly
        - If referencing visual information, explain it in detail
        - Cite which page or section the information comes from
        - If you cannot answer based on the provided context, say so clearly
        
        Answer:
        """
        
        # Generate answer using invoke method
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Extract source information
        sources = []
        for doc in relevant_docs:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "type": doc.metadata.get("type", "Unknown")
            }
            if source_info not in sources:
                sources.append(source_info)
        
        # Extract answer content (strip any leading/trailing whitespace)
        answer_content = response.content.strip()
        
        return {
            "answer": answer_content,
            "sources": sources,
            "context_used": context,
            "relevant_images": self.get_relevant_images(relevant_docs),
            "relevant_tables": self.extract_tables_from_context(relevant_docs)
        }
    
    def ingest_from_saved_extractions(self, extraction_directory: str = "document_extractions/content"):
        """Ingest documents from all saved extraction files"""
        print("Loading documents from saved extractions...")
        self.vector_store.add_documents_from_multiple_extractions(extraction_directory)
    
        return table_content
    
    # =============================================================================
    # SYSTEM MONITORING AND MANAGEMENT
    # =============================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the RAG system.
        
        Returns:
            Dictionary with extraction cache status, vector store statistics,
            and performance configuration details
        """
        print("Gathering system status information...")
        
        # Get extraction cache status
        cached_extractions = self.document_processor.get_all_saved_extractions()
        
        # Get vector store statistics
        vector_stats = self.vector_store.get_collection_statistics()
        
        # Get performance configuration
        performance_config = self.document_processor.get_performance_statistics()
        
        # Calculate totals
        total_cached_text = sum(ext['content_stats']['text_items'] for ext in cached_extractions)
        total_cached_tables = sum(ext['content_stats']['table_items'] for ext in cached_extractions)
        total_cached_visuals = sum(ext['content_stats']['visual_items'] for ext in cached_extractions)
        
        system_status = {
            "extraction_cache": {
                "total_cached_files": len(cached_extractions),
                "total_text_items": total_cached_text,
                "total_table_items": total_cached_tables,
                "total_visual_items": total_cached_visuals,
                "recent_extractions": sorted(
                    cached_extractions, 
                    key=lambda x: x['extraction_date'], 
                    reverse=True
                )[:5]  # Show 5 most recent
            },
            "vector_store": vector_stats,
            "performance_config": performance_config,
            "system_health": {
                "documents_in_vectors": vector_stats.get("total_documents", 0),
                "cached_extractions": len(cached_extractions),
                "optimization_level": self._calculate_optimization_level(performance_config)
            }
        }
        
        return system_status
    
    def display_system_status(self) -> None:
        """Display a formatted system status report."""
        status = self.get_system_status()
        
        print(f"\n{'='*60}")
        print(f"🤖 MULTI-MODAL RAG SYSTEM STATUS")
        print(f"{'='*60}")
        
        # Extraction Cache Status
        cache_info = status["extraction_cache"]
        print(f"\n📁 EXTRACTION CACHE:")
        print(f"   • Cached files: {cache_info['total_cached_files']}")
        print(f"   • Text items: {cache_info['total_text_items']}")
        print(f"   • Table items: {cache_info['total_table_items']}")
        print(f"   • Visual items: {cache_info['total_visual_items']}")
        
        # Vector Store Status
        vector_info = status["vector_store"]
        print(f"\n🗄️  VECTOR STORE:")
        print(f"   • Total documents: {vector_info.get('total_documents', 0)}")
        print(f"   • Text documents: {vector_info.get('collections', {}).get('text', {}).get('document_count', 0)}")
        print(f"   • Table documents: {vector_info.get('collections', {}).get('tables', {}).get('document_count', 0)}")
        print(f"   • Visual documents: {vector_info.get('collections', {}).get('visuals', {}).get('document_count', 0)}")
        
        # Performance Configuration
        perf_info = status["performance_config"]
        print(f"\n⚡ PERFORMANCE CONFIGURATION:")
        opt_status = perf_info.get("optimization_status", {})
        print(f"   • Batch Processing: {'✓' if opt_status.get('batch_processing_enabled') else '✗'}")
        print(f"   • Image Compression: {'✓' if opt_status.get('image_compression_enabled') else '✗'}")
        print(f"   • Smart Filtering: {'✓' if opt_status.get('smart_filtering_enabled') else '✗'}")
        print(f"   • Parallel Processing: {'✓' if opt_status.get('parallel_processing_enabled') else '✗'}")
        
        # System Health
        health_info = status["system_health"]
        print(f"\n🏥 SYSTEM HEALTH:")
        print(f"   • Optimization Level: {health_info['optimization_level']}")
        print(f"   • Ready for Queries: {'✓' if health_info['documents_in_vectors'] > 0 else '✗'}")
        
        print(f"\n{'='*60}")
    
    def _calculate_optimization_level(self, performance_config: Dict[str, Any]) -> str:
        """Calculate the optimization level based on enabled features."""
        opt_status = performance_config.get("optimization_status", {})
        
        enabled_optimizations = sum([
            opt_status.get('batch_processing_enabled', False),
            opt_status.get('image_compression_enabled', False),
            opt_status.get('smart_filtering_enabled', False),
            opt_status.get('parallel_processing_enabled', False),
            opt_status.get('duplicate_detection_enabled', False),
            opt_status.get('extraction_caching_enabled', False)
        ])
        
        if enabled_optimizations >= 5:
            return "Excellent (Most optimizations enabled)"
        elif enabled_optimizations >= 3:
            return "Good (Key optimizations enabled)"
        elif enabled_optimizations >= 1:
            return "Basic (Some optimizations enabled)"
        else:
            return "Minimal (Consider enabling optimizations)"
    
    def get_performance_recommendations(self) -> List[str]:
        """
        Get actionable performance recommendations based on current configuration.
        
        Returns:
            List of recommendation strings
        """
        performance_config = self.document_processor.get_performance_statistics()
        opt_status = performance_config.get("optimization_status", {})
        perf_settings = performance_config.get("performance_settings", {})
        
        recommendations = []
        
        if not opt_status.get('batch_processing_enabled', False):
            recommendations.append("Enable batch processing for 3x faster content extraction")
        
        if not opt_status.get('image_compression_enabled', False):
            recommendations.append("Enable image compression for 2x faster API uploads")
        
        if not opt_status.get('smart_filtering_enabled', False):
            recommendations.append("Enable smart filtering to automatically skip low-content pages")
        
        if not opt_status.get('parallel_processing_enabled', False):
            recommendations.append("Enable parallel processing for faster large document handling")
        
        if perf_settings.get('max_workers', 1) < 4:
            recommendations.append("Consider increasing MAX_WORKERS to 4-6 for better parallelization")
        
        if perf_settings.get('image_quality', 100) > 85:
            recommendations.append("Consider reducing IMAGE_QUALITY to 85% for faster processing with minimal quality loss")
        
        if not recommendations:
            recommendations.append("Your system is well-optimized! Consider monitoring performance over time.")
        
        return recommendations
    
    def clear_extraction_cache(self, keep_recent_days: int = 30) -> None:
        """
        Clear old extraction cache files to free up disk space.
        
        Args:
            keep_recent_days: Number of recent days to keep in cache
        """
        print(f"Clearing extraction cache (keeping last {keep_recent_days} days)...")
        
        # This would be implemented in document_processor if the method exists
        if hasattr(self.document_processor, 'clear_saved_extractions'):
            self.document_processor.clear_saved_extractions(keep_recent_days)
        else:
            print("Cache clearing not implemented in document processor")
    
    def force_reprocess_document(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Force reprocessing of a document, bypassing all caches.
        
        Args:
            file_path: Path to document to reprocess
            
        Returns:
            Processed content dictionary
        """
        print(f"Force reprocessing document: {os.path.basename(file_path)}")
        return self.ingest_single_document(file_path, force_reprocess=True)
    
    # =============================================================================
    # LEGACY METHOD SUPPORT (for backward compatibility)
    # =============================================================================
    
    def ingest_document(self, file_path: str, force_reprocess: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method name - use ingest_single_document() instead."""
        return self.ingest_single_document(file_path, force_reprocess)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Legacy method name - use generate_answer() instead."""
        return self.generate_answer(question)
    
    def ingest_from_saved_extractions(self, extraction_directory: str = "document_extractions/content") -> Tuple[int, int]:
        """Legacy method name - use load_from_cached_extractions() instead."""
        return self.load_from_cached_extractions(extraction_directory)
    
    def ingest_folder_with_caching(self, folder_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Legacy method name - use ingest_folder_of_documents() instead."""
        return self.ingest_folder_of_documents(folder_path, force_reprocess)
    
    def get_extraction_status(self) -> List[Dict[str, Any]]:
        """Legacy method name - use get_system_status() instead."""
        return self.get_system_status()["extraction_cache"]["recent_extractions"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Legacy method name - use get_system_status() instead."""
        return self.document_processor.get_performance_statistics()