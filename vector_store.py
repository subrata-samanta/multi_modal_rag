"""
Multi-Modal Vector Store for RAG Pipeline

This module provides a comprehensive vector storage system that handles different
content types (text, tables, visuals) separately for optimal retrieval performance.
Features intelligent query analysis and context-aware document retrieval.


"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from config import Config


class MultiModalVectorStore:
    """
    Advanced vector store for multi-modal document retrieval.
    
    Manages separate collections for text, tables, and visual content,
    with intelligent query analysis for optimal context retrieval.
    """
    
    def __init__(self):
        """Initialize the multi-modal vector store with separate collections."""
        print("Initializing Multi-Modal Vector Store...")
        
        # Setup core components
        self._initialize_embedding_system()
        self._initialize_query_analysis_system()
        self._setup_text_processing()
        self._create_specialized_collections()
        
        print("✓ Multi-Modal Vector Store initialized successfully")
    
    # =============================================================================
    # INITIALIZATION AND SETUP
    # =============================================================================
    
    def _initialize_embedding_system(self) -> None:
        """
        Initialize the embedding system with Google Vertex AI.
        
        Sets up high-quality text embeddings for semantic similarity search.
        """
        try:
            # Configure Google Cloud credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            # Initialize embeddings with latest model
            self.embeddings = VertexAIEmbeddings(
                model_name="google-text-embedding-004",
                project=Config.GOOGLE_PROJECT_ID,
                location=Config.GOOGLE_LOCATION
            )
            
            print("✓ Embedding system initialized")
            
        except Exception as e:
            raise Exception(f"Failed to initialize embedding system: {e}")
    
    def _initialize_query_analysis_system(self) -> None:
        """
        Initialize the AI system for intelligent query analysis.
        
        Uses Gemini Pro for understanding query intent and optimizing retrieval.
        """
        try:
            # Ensure credentials are set
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            # Initialize query analyzer with low temperature for consistent analysis
            self.query_analyzer = ChatVertexAI(
                model_name="gemini-pro",
                temperature=0.1,
                project=Config.GOOGLE_PROJECT_ID,
                location=Config.GOOGLE_LOCATION
            )
            
            print("✓ Query analysis system initialized")
            
        except Exception as e:
            raise Exception(f"Failed to initialize query analysis system: {e}")
    
    def _setup_text_processing(self) -> None:
        """Configure text splitting for optimal chunk sizes."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        print("✓ Text processing configured")
    
    def _create_specialized_collections(self) -> None:
        """
        Create separate vector collections for different content types.
        
        This separation allows for optimized retrieval strategies per content type.
        """
        # Text content collection - for paragraphs, explanations, general content
        self.text_collection = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_text",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/text"
        )
        
        # Table content collection - for structured data, statistics, comparisons  
        self.table_collection = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_tables",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/tables"
        )
        
        # Visual content collection - for charts, diagrams, visual descriptions
        self.visual_collection = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_visuals",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/visuals"
        )
        
        print("✓ Specialized collections created")
    
    
    # =============================================================================
    # DOCUMENT STORAGE OPERATIONS
    # =============================================================================
    
    def store_processed_documents(self, processed_content: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Store processed documents in their respective specialized collections.
        
        Args:
            processed_content: Dictionary containing text, tables, and visuals lists
        """
        print("Storing documents in vector collections...")
        
        # Store text documents with intelligent chunking
        text_documents_stored = self._store_text_documents(processed_content.get("text", []))
        
        # Store table documents (usually don't need chunking)
        table_documents_stored = self._store_table_documents(processed_content.get("tables", []))
        
        # Store visual documents with descriptive content
        visual_documents_stored = self._store_visual_documents(processed_content.get("visuals", []))
        
        total_stored = text_documents_stored + table_documents_stored + visual_documents_stored
        print(f"✓ Successfully stored {total_stored} documents across all collections")
    
    def _store_text_documents(self, text_items: List[Dict[str, Any]]) -> int:
        """
        Store text content with intelligent chunking for optimal retrieval.
        
        Returns:
            Number of document chunks stored
        """
        if not text_items:
            return 0
        
        text_documents = []
        
        for item in text_items:
            # Split large text content into manageable chunks
            content_chunks = self.text_splitter.split_text(item["content"])
            
            # Create document for each chunk with preserved metadata
            for chunk_index, chunk_content in enumerate(content_chunks):
                document = Document(
                    page_content=chunk_content,
                    metadata={
                        "source_file": item["source"],
                        "page_number": item["page"],
                        "content_type": item["type"],
                        "image_reference": item.get("image_path", ""),
                        "chunk_index": chunk_index,
                        "total_chunks": len(content_chunks),
                        "storage_timestamp": datetime.now().isoformat()
                    }
                )
                text_documents.append(document)
        
        if text_documents:
            self.text_collection.add_documents(text_documents)
            print(f"✓ Stored {len(text_documents)} text document chunks")
        
        return len(text_documents)
    
    def _store_table_documents(self, table_items: List[Dict[str, Any]]) -> int:
        """
        Store table content as complete units (tables usually shouldn't be chunked).
        
        Returns:
            Number of table documents stored
        """
        if not table_items:
            return 0
        
        table_documents = []
        
        for item in table_items:
            document = Document(
                page_content=item["content"],
                metadata={
                    "source_file": item["source"],
                    "page_number": item["page"],
                    "content_type": item["type"],
                    "image_reference": item.get("image_path", ""),
                    "data_structure": "table",
                    "storage_timestamp": datetime.now().isoformat()
                }
            )
            table_documents.append(document)
        
        if table_documents:
            self.table_collection.add_documents(table_documents)
            print(f"✓ Stored {len(table_documents)} table documents")
        
        return len(table_documents)
    
    def _store_visual_documents(self, visual_items: List[Dict[str, Any]]) -> int:
        """
        Store visual content descriptions for semantic search.
        
        Returns:
            Number of visual documents stored
        """
        if not visual_items:
            return 0
        
        visual_documents = []
        
        for item in visual_items:
            document = Document(
                page_content=item["content"],
                metadata={
                    "source_file": item["source"],
                    "page_number": item["page"],
                    "content_type": item["type"],
                    "image_reference": item.get("image_path", ""),
                    "visual_type": "chart_diagram_visual",
                    "storage_timestamp": datetime.now().isoformat()
                }
            )
            visual_documents.append(document)
        
        if visual_documents:
            self.visual_collection.add_documents(visual_documents)
            print(f"✓ Stored {len(visual_documents)} visual documents")
        
        return len(visual_documents)
    
    
    # =============================================================================
    # BATCH LOADING FROM EXTRACTION FILES
    # =============================================================================
    
    def load_documents_from_extraction_file(self, extraction_file_path: str) -> bool:
        """
        Load and store documents from a single extraction file.
        
        Args:
            extraction_file_path: Path to the JSON extraction file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(extraction_file_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            # Extract content and metadata
            processed_content = extraction_data.get("content", {})
            source_file = extraction_data.get("source_file", "unknown")
            
            print(f"Loading extraction: {os.path.basename(source_file)}")
            
            # Store documents in vector collections
            self.store_processed_documents(processed_content)
            
            return True
            
        except Exception as e:
            print(f"Error loading extraction file {extraction_file_path}: {e}")
            return False
    
    def bulk_load_from_extraction_directory(self, extraction_directory: str = "document_extractions/content") -> Tuple[int, int]:
        """
        Load documents from all extraction files in a directory.
        
        Args:
            extraction_directory: Directory containing extraction JSON files
            
        Returns:
            Tuple of (successful_loads, total_files)
        """
        if not os.path.exists(extraction_directory):
            print(f"Extraction directory not found: {extraction_directory}")
            return (0, 0)
        
        # Find all extraction files
        extraction_files = [f for f in os.listdir(extraction_directory) if f.endswith('.json')]
        
        if not extraction_files:
            print("No extraction files found in directory")
            return (0, 0)
        
        print(f"Found {len(extraction_files)} extraction files to load")
        
        # Load each file
        successful_loads = 0
        for filename in extraction_files:
            filepath = os.path.join(extraction_directory, filename)
            if self.load_documents_from_extraction_file(filepath):
                successful_loads += 1
                print(f"✓ Loaded: {filename}")
            else:
                print(f"✗ Failed: {filename}")
        
        print(f"✓ Bulk loading complete: {successful_loads}/{len(extraction_files)} files loaded")
        return (successful_loads, len(extraction_files))
    
    
    # =============================================================================
    # INTELLIGENT QUERY ANALYSIS AND RETRIEVAL
    # =============================================================================
    
    def search_relevant_content(self, query: str, max_results_per_type: int = 3) -> Dict[str, List[Document]]:
        """
        Search across all content types with balanced retrieval.
        
        Args:
            query: User search query
            max_results_per_type: Maximum results to return per content type
            
        Returns:
            Dictionary with results from each content type
        """
        print(f"Searching for: '{query}'")
        
        results = {
            "text": self.text_collection.similarity_search(query, k=max_results_per_type),
            "tables": self.table_collection.similarity_search(query, k=max_results_per_type),
            "visuals": self.visual_collection.similarity_search(query, k=max_results_per_type)
        }
        
        total_results = sum(len(docs) for docs in results.values())
        print(f"Found {total_results} relevant documents across all types")
        
        return results
    
    def get_contextually_relevant_documents(self, query: str) -> List[Document]:
        """
        Get contextually relevant documents using AI-driven retrieval strategy.
        
        Uses LLM to analyze query intent and optimize retrieval weights for
        different content types based on what would best answer the question.
        
        Args:
            query: User query to analyze and search
            
        Returns:
            List of most relevant documents across all content types
        """
        print(f"Analyzing query intent for: '{query}'")
        
        # Use AI to determine optimal retrieval strategy
        content_type_weights = self._analyze_query_intent_with_ai(query)
        
        # Convert weights to specific retrieval counts
        retrieval_counts = self._calculate_retrieval_distribution(content_type_weights)
        
        print(f"AI-driven retrieval strategy: {retrieval_counts}")
        
        # Retrieve documents according to AI-determined strategy
        search_results = {
            "text": self.text_collection.similarity_search(query, k=retrieval_counts["text"]),
            "tables": self.table_collection.similarity_search(query, k=retrieval_counts["tables"]),
            "visuals": self.visual_collection.similarity_search(query, k=retrieval_counts["visuals"])
        }
        
        # Combine all relevant documents
        all_relevant_documents = []
        for content_type, documents in search_results.items():
            all_relevant_documents.extend(documents)
        
        print(f"Retrieved {len(all_relevant_documents)} contextually relevant documents")
        return all_relevant_documents
    
    def _analyze_query_intent_with_ai(self, query: str) -> Dict[str, int]:
        """
        Use AI to analyze query intent and determine optimal content type weights.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with weights (1-5) for each content type
        """
        analysis_prompt = f"""
        You are an expert at analyzing user queries to determine the best retrieval strategy
        for a multi-modal document database.
        
        User Query: "{query}"
        
        Available content types:
        - TEXT: General paragraphs, explanations, narrative content, definitions
        - TABLES: Structured data, statistics, numerical comparisons, data tables
        - VISUALS: Charts, graphs, diagrams, visual elements, figure descriptions
        
        Analyze this query and assign retrieval weights (1-5 scale) based on:
        1. What type of information would best answer this question?
        2. What content format would be most useful?
        3. How much emphasis should be placed on each content type?
        
        Weight Guidelines:
        - 5: Critical for answering the query
        - 4: Very important 
        - 3: Moderately important
        - 2: Somewhat relevant
        - 1: Minimally relevant
        
        Respond ONLY in this exact format:
        text: [1-5]
        tables: [1-5]
        visuals: [1-5]
        reasoning: [brief explanation of your analysis]
        
        Example Response:
        text: 2
        tables: 5
        visuals: 3
        reasoning: Query asks for specific numerical comparison best found in tables, with some visual support helpful
        """
        
        try:
            ai_response = self.query_analyzer.invoke([HumanMessage(content=analysis_prompt)])
            return self._parse_ai_analysis_response(ai_response.content)
            
        except Exception as e:
            print(f"Error in AI query analysis: {e}")
            # Fallback to balanced retrieval
            return {"text": 3, "tables": 3, "visuals": 2}
    
    def _parse_ai_analysis_response(self, ai_response: str) -> Dict[str, int]:
        """
        Parse AI response to extract content type weights.
        
        Args:
            ai_response: Raw response from AI analysis
            
        Returns:
            Dictionary with parsed weights, defaults if parsing fails
        """
        # Default balanced weights
        weights = {"text": 3, "tables": 3, "visuals": 2}
        
        try:
            response_lines = ai_response.strip().split('\n')
            
            for line in response_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    
                    if key in weights:
                        try:
                            # Extract numeric value and clamp to valid range
                            weight_value = int(value.strip())
                            weights[key] = max(1, min(5, weight_value))
                        except ValueError:
                            continue  # Skip invalid numeric values
            
            # Log the reasoning if available
            for line in response_lines:
                if line.strip().lower().startswith('reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
                    print(f"AI reasoning: {reasoning}")
                    break
                    
        except Exception as e:
            print(f"Error parsing AI analysis response: {e}")
        
        return weights
    
    def _calculate_retrieval_distribution(self, weights: Dict[str, int], max_total_documents: int = 9) -> Dict[str, int]:
        """
        Convert content type weights to specific document retrieval counts.
        
        Args:
            weights: Weight values for each content type
            max_total_documents: Maximum total documents to retrieve
            
        Returns:
            Dictionary with specific retrieval counts per content type
        """
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            # Fallback if all weights are zero
            return {"text": 3, "tables": 3, "visuals": 3}
        
        # Calculate proportional distribution
        distribution = {}
        for content_type, weight in weights.items():
            count = max(1, int((weight / total_weight) * max_total_documents))
            distribution[content_type] = count
        
        return distribution
    
    # =============================================================================
    # COLLECTION MANAGEMENT AND UTILITIES
    # =============================================================================
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all vector collections.
        
        Returns:
            Dictionary with document counts and collection health info
        """
        try:
            stats = {
                "collections": {
                    "text": {
                        "document_count": self.text_collection._collection.count(),
                        "collection_name": self.text_collection._collection.name
                    },
                    "tables": {
                        "document_count": self.table_collection._collection.count(),
                        "collection_name": self.table_collection._collection.name
                    },
                    "visuals": {
                        "document_count": self.visual_collection._collection.count(),
                        "collection_name": self.visual_collection._collection.name
                    }
                },
                "total_documents": (
                    self.text_collection._collection.count() +
                    self.table_collection._collection.count() +
                    self.visual_collection._collection.count()
                ),
                "persist_directory": Config.PERSIST_DIRECTORY,
                "embedding_model": "google-text-embedding-004"
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting collection statistics: {e}")
            return {"error": str(e)}
    
    def clear_all_collections(self) -> bool:
        """
        Clear all documents from all collections.
        
        WARNING: This will permanently delete all stored documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("⚠️  Clearing all vector collections...")
            
            # Clear each collection
            self.text_collection.delete_collection()
            self.table_collection.delete_collection()
            self.visual_collection.delete_collection()
            
            # Recreate empty collections
            self._create_specialized_collections()
            
            print("✓ All collections cleared and recreated")
            return True
            
        except Exception as e:
            print(f"Error clearing collections: {e}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], max_results: int = 10) -> List[Document]:
        """
        Search documents by metadata criteria across all collections.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of documents matching the metadata criteria
        """
        try:
            all_matching_documents = []
            
            # Search each collection with metadata filter
            for collection_name, collection in [
                ("text", self.text_collection),
                ("tables", self.table_collection),
                ("visuals", self.visual_collection)
            ]:
                try:
                    # Chroma metadata filtering (exact match)
                    results = collection.get(
                        where=metadata_filter,
                        limit=max_results
                    )
                    
                    # Convert to Document objects
                    for i, doc_id in enumerate(results['ids']):
                        doc = Document(
                            page_content=results['documents'][i],
                            metadata=results['metadatas'][i]
                        )
                        all_matching_documents.append(doc)
                        
                except Exception as e:
                    print(f"Error searching {collection_name} collection: {e}")
                    continue
            
            print(f"Found {len(all_matching_documents)} documents matching metadata filter")
            return all_matching_documents[:max_results]
            
        except Exception as e:
            print(f"Error in metadata search: {e}")
            return []
    
    def get_document_sources(self) -> List[str]:
        """
        Get a list of all unique document sources in the vector store.
        
        Returns:
            List of unique source file paths
        """
        try:
            all_sources = set()
            
            # Get sources from each collection
            for collection in [self.text_collection, self.table_collection, self.visual_collection]:
                try:
                    collection_data = collection.get()
                    for metadata in collection_data['metadatas']:
                        if 'source_file' in metadata:
                            all_sources.add(metadata['source_file'])
                except Exception as e:
                    print(f"Error getting sources from collection: {e}")
                    continue
            
            return sorted(list(all_sources))
            
        except Exception as e:
            print(f"Error getting document sources: {e}")
            return []
    
    # =============================================================================
    # LEGACY METHOD SUPPORT (for backward compatibility)
    # =============================================================================
    
    def add_documents(self, processed_content: Dict[str, List[Dict[str, Any]]]) -> None:
        """Legacy method name - use store_processed_documents() instead."""
        return self.store_processed_documents(processed_content)
    
    def add_documents_from_extraction_file(self, extraction_file_path: str) -> bool:
        """Legacy method name - use load_documents_from_extraction_file() instead."""
        return self.load_documents_from_extraction_file(extraction_file_path)
    
    def add_documents_from_multiple_extractions(self, extraction_directory: str = "document_extractions/content") -> Tuple[int, int]:
        """Legacy method name - use bulk_load_from_extraction_directory() instead."""
        return self.bulk_load_from_extraction_directory(extraction_directory)
    
    def get_contextual_documents(self, query: str) -> List[Document]:
        """Legacy method name - use get_contextually_relevant_documents() instead."""
        return self.get_contextually_relevant_documents(query)
