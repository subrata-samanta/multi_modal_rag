"""
Multi-Modal RAG Pipeline - Interactive Command Line Interface

A user-friendly command line interface for the Multi-Modal RAG Pipeline.
Provides comprehensive document ingestion and intelligent question-answering
capabilities with support for PDF, PPTX, EML, and Excel files.

Features:
- Interactive document ingestion with progress tracking
- Intelligent question answering with multiple modes
- System monitoring and performance analytics
- Comprehensive logging and error handling

"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from rag_pipeline import MultiModalRAGPipeline


class RAGInterface:
    """
    Interactive command-line interface for the Multi-Modal RAG Pipeline.
    
    Provides a clean, user-friendly interface for document processing
    and question-answering operations.
    """
    
    def __init__(self):
        """Initialize the RAG interface and pipeline."""
        print("🚀 Initializing Multi-Modal RAG Pipeline...")
        
        try:
            self.rag_pipeline = MultiModalRAGPipeline()
            print("✅ RAG Pipeline initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG Pipeline: {e}")
            sys.exit(1)
        
        # Setup logging directory
        self.log_directory = Path("extraction_logs")
        self.log_directory.mkdir(exist_ok=True)
    
    def display_welcome_message(self) -> None:
        """Display the welcome message and system capabilities."""
        print("\n" + "="*70)
        print("🤖 MULTI-MODAL RAG PIPELINE - INTERACTIVE INTERFACE")
        print("="*70)
        print("📋 SUPPORTED FORMATS:")
        print("   • PDF documents with text, tables, and images")
        print("   • PowerPoint presentations (PPTX)")
        print("   • Email files (EML) with attachments")
        print("   • Excel spreadsheets (XLS/XLSX) with embedded images")
        print("\n🧠 AI CAPABILITIES:")
        print("   • Intelligent content extraction using Google Gemini")
        print("   • Multi-modal understanding (text, tables, visuals)")
        print("   • Context-aware question answering")
        print("   • Performance optimization and caching")
        print("="*70)
    
    def display_main_menu(self) -> None:
        """Display the main menu options."""
        print("\n" + "="*50)
        print("📚 MAIN MENU")
        print("="*50)
        print("1. 📄 Document Management")
        print("2. 🤔 Ask Questions")
        print("3. 📊 System Status")
        print("4. ⚙️  Settings & Performance")
        print("5. 🚪 Exit")
        print("="*50)
    
    def run(self) -> None:
        """Run the main interface loop."""
        self.display_welcome_message()
        
        while True:
            self.display_main_menu()
            
            try:
                choice = input("\n🎯 Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    self.handle_document_management()
                elif choice == "2":
                    self.handle_question_answering()
                elif choice == "3":
                    self.handle_system_status()
                elif choice == "4":
                    self.handle_settings_and_performance()
                elif choice == "5":
                    self.handle_exit()
                    break
                else:
                    print("❌ Invalid choice. Please enter a number between 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again.")
    
    # =============================================================================
    # DOCUMENT MANAGEMENT OPERATIONS
    # =============================================================================
    
    def handle_document_management(self) -> None:
        """Handle all document management operations."""
        print("\n" + "="*50)
        print("📄 DOCUMENT MANAGEMENT")
        print("="*50)
        print("1. 📁 Ingest Single Document")
        print("2. 📂 Ingest Folder of Documents")
        print("3. 💾 Load from Cached Extractions")
        print("4. 🗂️  View Extraction Cache")
        print("5. 🔄 Force Reprocess Document")
        print("6. 🧹 Clear Old Cache Files")
        print("7. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Enter your choice (1-7): ").strip()
        
        if choice == "1":
            self.ingest_single_document()
        elif choice == "2":
            self.ingest_document_folder()
        elif choice == "3":
            self.load_cached_extractions()
        elif choice == "4":
            self.view_extraction_cache()
        elif choice == "5":
            self.force_reprocess_document()
        elif choice == "6":
            self.clear_old_cache()
        elif choice == "7":
            return
        else:
            print("❌ Invalid choice. Please try again.")
    
    def ingest_single_document(self) -> None:
        """Handle single document ingestion with comprehensive feedback."""
        print("\n📄 SINGLE DOCUMENT INGESTION")
        print("-" * 40)
        
        file_path = input("📁 Enter document path: ").strip().strip('"\'')
        
        # Validate file existence
        if not os.path.exists(file_path):
            print("❌ File not found. Please check the path and try again.")
            return
        
        # Validate file type
        supported_extensions = ['.pdf', '.pptx', '.eml', '.xls', '.xlsx']
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in supported_extensions:
            print(f"❌ Unsupported file type: {file_extension}")
            print(f"📋 Supported formats: {', '.join(supported_extensions)}")
            return
        
        # Check for cached extraction
        force_reprocess = False
        if hasattr(self.rag_pipeline.document_processor, 'has_valid_cache'):
            if self.rag_pipeline.document_processor.has_valid_cache(file_path):
                use_cache = input("💾 Found cached extraction. Use cache? (y/n, default: y): ").strip().lower()
                force_reprocess = use_cache == 'n'
        
        # Process document
        try:
            print(f"\n🔄 Processing: {Path(file_path).name}")
            print("⏳ Please wait...")
            
            start_time = datetime.now()
            processed_content = self.rag_pipeline.ingest_single_document(file_path, force_reprocess)
            end_time = datetime.now()
            
            # Display results
            processing_time = (end_time - start_time).total_seconds()
            self._display_ingestion_results(processed_content, processing_time, Path(file_path).name)
            
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            self._save_error_log(file_path, str(e))
    
    def ingest_document_folder(self) -> None:
        """Handle folder ingestion with progress tracking."""
        print("\n📂 FOLDER INGESTION")
        print("-" * 40)
        
        folder_path = input("📁 Enter folder path: ").strip().strip('"\'')
        
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print("❌ Invalid folder path. Please check and try again.")
            return
        
        force_reprocess = input("🔄 Force reprocess all files? (y/n, default: n): ").strip().lower() == 'y'
        
        try:
            print(f"\n🔄 Processing folder: {Path(folder_path).name}")
            print("⏳ Please wait...")
            
            start_time = datetime.now()
            results = self.rag_pipeline.ingest_folder_of_documents(folder_path, force_reprocess)
            end_time = datetime.now()
            
            # Display folder processing results
            processing_time = (end_time - start_time).total_seconds()
            self._display_folder_results(results, processing_time)
            
        except Exception as e:
            print(f"❌ Error processing folder: {e}")
    
    def _display_ingestion_results(self, processed_content: Dict[str, List], processing_time: float, filename: str) -> None:
        """Display comprehensive ingestion results."""
        text_count = len(processed_content.get('text', []))
        table_count = len(processed_content.get('tables', []))
        visual_count = len(processed_content.get('visuals', []))
        total_items = text_count + table_count + visual_count
        
        print(f"\n✅ Successfully processed: {filename}")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Extraction summary:")
        print(f"   • Text items: {text_count}")
        print(f"   • Table items: {table_count}")
        print(f"   • Visual items: {visual_count}")
        print(f"   • Total items: {total_items}")
        
        if total_items > 0:
            print(f"🎯 Average time per item: {processing_time/total_items:.2f} seconds")
        
        print("✅ Document successfully added to knowledge base!")
    
    def _display_folder_results(self, results: Dict[str, Any], processing_time: float) -> None:
        """Display folder processing results."""
        successful_files = results.get('successful_files', 0)
        total_files = results.get('total_files', 0)
        total_documents = results.get('total_documents', 0)
        
        print(f"\n✅ Folder processing complete!")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📁 Files processed: {successful_files}/{total_files}")
        print(f"📊 Total documents stored: {total_documents}")
        
        if successful_files > 0:
            print(f"🎯 Average time per file: {processing_time/successful_files:.2f} seconds")
    
    def load_cached_extractions(self) -> None:
        """Load documents from cached extraction files."""
        print("\n💾 LOAD CACHED EXTRACTIONS")
        print("-" * 40)
        
        extraction_dir = input("📁 Enter extraction directory (default: document_extractions/content): ").strip()
        if not extraction_dir:
            extraction_dir = "document_extractions/content"
        
        try:
            print("🔄 Loading cached extractions...")
            successful_loads, total_files = self.rag_pipeline.load_from_cached_extractions(extraction_dir)
            
            print(f"✅ Loaded {successful_loads}/{total_files} extraction files")
            
        except Exception as e:
            print(f"❌ Error loading cached extractions: {e}")
    
    def view_extraction_cache(self) -> None:
        """Display extraction cache status."""
        print("\n🗂️  EXTRACTION CACHE STATUS")
        print("-" * 40)
        
        try:
            self.rag_pipeline.display_system_status()
        except Exception as e:
            print(f"❌ Error retrieving cache status: {e}")
    
    def force_reprocess_document(self) -> None:
        """Force reprocess a document, bypassing cache."""
        print("\n🔄 FORCE REPROCESS DOCUMENT")
        print("-" * 40)
        
        file_path = input("📁 Enter document path to reprocess: ").strip().strip('"\'')
        
        if not os.path.exists(file_path):
            print("❌ File not found. Please check the path.")
            return
        
        try:
            print(f"🔄 Force reprocessing: {Path(file_path).name}")
            print("⏳ Please wait...")
            
            start_time = datetime.now()
            processed_content = self.rag_pipeline.force_reprocess_document(file_path)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            self._display_ingestion_results(processed_content, processing_time, Path(file_path).name)
            
        except Exception as e:
            print(f"❌ Error reprocessing document: {e}")
    
    def clear_old_cache(self) -> None:
        """Clear old cache files to free up space."""
        print("\n🧹 CLEAR OLD CACHE FILES")
        print("-" * 40)
        
        days_input = input("📅 Keep files from last N days (default: 30): ").strip()
        
        try:
            keep_days = int(days_input) if days_input else 30
            print(f"🧹 Clearing cache files older than {keep_days} days...")
            self.rag_pipeline.clear_extraction_cache(keep_days)
            
        except ValueError:
            print("❌ Invalid number. Using default (30 days).")
            self.rag_pipeline.clear_extraction_cache(30)
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    
    # =============================================================================
    # QUESTION ANSWERING OPERATIONS
    # =============================================================================
    
    def handle_question_answering(self) -> None:
        """Handle intelligent question answering with multiple modes."""
        print("\n" + "="*50)
        print("🤔 QUESTION ANSWERING")
        print("="*50)
        print("1. 🎯 Smart Question (AI-optimized retrieval)")
        print("2. 📊 Quick Stats (System information)")
        print("3. 🔍 Search Documents (Find specific content)")
        print("4. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            self.smart_question_answering()
        elif choice == "2":
            self.quick_system_stats()
        elif choice == "3":
            self.search_documents()
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice. Please try again.")
    
    def smart_question_answering(self) -> None:
        """Handle intelligent question answering with context-aware retrieval."""
        print("\n🎯 SMART QUESTION ANSWERING")
        print("-" * 40)
        print("💡 Tip: Ask questions about your ingested documents.")
        print("   Examples:")
        print("   • 'What are the key findings in the financial report?'")
        print("   • 'Show me sales data from the presentation'")
        print("   • 'What charts or graphs are mentioned?'")
        
        question = input("\n❓ Enter your question: ").strip()
        
        if not question:
            print("❌ Please enter a valid question.")
            return
        
        try:
            print("\n🧠 Analyzing question and retrieving relevant information...")
            print("⏳ Please wait...")
            
            start_time = datetime.now()
            result = self.rag_pipeline.generate_answer(question)
            end_time = datetime.now()
            
            # Display comprehensive answer
            self._display_answer_results(result, end_time - start_time)
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    def _display_answer_results(self, result: Dict[str, Any], processing_time: datetime) -> None:
        """Display comprehensive answer results with sources and context."""
        print(f"\n" + "="*60)
        print("🎯 ANSWER")
        print("="*60)
        print(result["answer"])
        
        # Display sources
        sources = result.get("sources", [])
        if sources:
            print(f"\n📚 SOURCES ({len(sources)} references)")
            print("-" * 30)
            for i, source in enumerate(sources, 1):
                source_name = source.get("source_file", "Unknown")
                page = source.get("page_number", "Unknown")
                content_type = source.get("content_type", "Unknown")
                print(f"{i}. {source_name} (Page {page}, {content_type})")
        
        # Display supporting materials
        supporting_tables = result.get("supporting_tables", [])
        if supporting_tables:
            print(f"\n📊 SUPPORTING TABLES ({len(supporting_tables)} found)")
            print("-" * 30)
            for i, table in enumerate(supporting_tables, 1):
                source_name = table.get("source_file", "Unknown")
                page = table.get("page_number", "Unknown")
                print(f"{i}. Table from {source_name} (Page {page})")
        
        supporting_images = result.get("supporting_images", [])
        if supporting_images:
            print(f"\n🖼️  SUPPORTING VISUALS ({len(supporting_images)} found)")
            print("-" * 30)
            for i, image in enumerate(supporting_images, 1):
                source_name = image.get("source_file", "Unknown")
                page = image.get("page_number", "Unknown")
                print(f"{i}. Visual from {source_name} (Page {page})")
        
        # Display metadata
        confidence = result.get("confidence", "unknown")
        print(f"\n📈 METADATA")
        print("-" * 30)
        print(f"• Confidence: {confidence}")
        print(f"• Processing time: {processing_time.total_seconds():.2f} seconds")
        print(f"• Sources used: {len(sources)}")
        
        print("="*60)
    
    def quick_system_stats(self) -> None:
        """Display quick system statistics."""
        print("\n📊 QUICK SYSTEM STATISTICS")
        print("-" * 40)
        
        try:
            # Get vector store statistics
            vector_stats = self.rag_pipeline.vector_store.get_collection_statistics()
            
            print("🗄️  Vector Store Status:")
            print(f"   • Total documents: {vector_stats.get('total_documents', 0)}")
            
            collections = vector_stats.get('collections', {})
            if collections:
                for coll_type, coll_info in collections.items():
                    count = coll_info.get('document_count', 0)
                    print(f"   • {coll_type.title()} documents: {count}")
            
            # Get document sources
            sources = self.rag_pipeline.vector_store.get_document_sources()
            print(f"\n📁 Document Sources ({len(sources)} unique files):")
            for source in sources[:5]:  # Show first 5
                print(f"   • {Path(source).name}")
            
            if len(sources) > 5:
                print(f"   • ... and {len(sources) - 5} more files")
            
        except Exception as e:
            print(f"❌ Error retrieving system stats: {e}")
    
    def search_documents(self) -> None:
        """Search for specific content across documents."""
        print("\n🔍 SEARCH DOCUMENTS")
        print("-" * 40)
        
        search_query = input("🔍 Enter search query: ").strip()
        
        if not search_query:
            print("❌ Please enter a valid search query.")
            return
        
        try:
            print(f"\n🔍 Searching for: '{search_query}'")
            print("⏳ Please wait...")
            
            # Use the vector store search functionality
            search_results = self.rag_pipeline.vector_store.search_relevant_content(search_query, max_results_per_type=5)
            
            # Display search results
            total_results = sum(len(docs) for docs in search_results.values())
            
            if total_results == 0:
                print("❌ No documents found matching your search query.")
                return
            
            print(f"\n✅ Found {total_results} relevant documents")
            
            for content_type, documents in search_results.items():
                if documents:
                    print(f"\n📝 {content_type.upper()} RESULTS ({len(documents)} found):")
                    print("-" * 30)
                    
                    for i, doc in enumerate(documents, 1):
                        source = Path(doc.metadata.get("source_file", "Unknown")).name
                        page = doc.metadata.get("page_number", "Unknown")
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        
                        print(f"{i}. {source} (Page {page})")
                        print(f"   Preview: {preview}")
                        print()
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
    
    # =============================================================================
    # SYSTEM STATUS AND MONITORING
    # =============================================================================
    
    def handle_system_status(self) -> None:
        """Handle system status and monitoring operations."""
        print("\n" + "="*50)
        print("📊 SYSTEM STATUS")
        print("="*50)
        print("1. 🏥 Overall System Health")
        print("2. 📈 Performance Statistics")
        print("3. 🗄️  Vector Store Details")
        print("4. 💾 Cache Information")
        print("5. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Enter your choice (1-5): ").strip()
        
        if choice == "1":
            self.display_system_health()
        elif choice == "2":
            self.display_performance_stats()
        elif choice == "3":
            self.display_vector_store_details()
        elif choice == "4":
            self.display_cache_information()
        elif choice == "5":
            return
        else:
            print("❌ Invalid choice. Please try again.")
    
    def display_system_health(self) -> None:
        """Display comprehensive system health information."""
        print("\n🏥 SYSTEM HEALTH CHECK")
        print("-" * 40)
        
        try:
            self.rag_pipeline.display_system_status()
        except Exception as e:
            print(f"❌ Error retrieving system health: {e}")
    
    def display_performance_stats(self) -> None:
        """Display detailed performance statistics."""
        print("\n📈 PERFORMANCE STATISTICS")
        print("-" * 40)
        
        try:
            performance_stats = self.rag_pipeline.document_processor.get_performance_statistics()
            
            # Display optimization status
            opt_status = performance_stats.get("optimization_status", {})
            print("⚡ OPTIMIZATION STATUS:")
            optimizations = [
                ("Batch Processing", opt_status.get('batch_processing_enabled', False)),
                ("Image Compression", opt_status.get('image_compression_enabled', False)),
                ("Smart Filtering", opt_status.get('smart_filtering_enabled', False)),
                ("Parallel Processing", opt_status.get('parallel_processing_enabled', False)),
                ("Duplicate Detection", opt_status.get('duplicate_detection_enabled', False)),
                ("Extraction Caching", opt_status.get('extraction_caching_enabled', False))
            ]
            
            for opt_name, enabled in optimizations:
                status = "✅ Enabled" if enabled else "❌ Disabled"
                print(f"   • {opt_name}: {status}")
            
            # Display performance settings
            perf_settings = performance_stats.get("performance_settings", {})
            print(f"\n⚙️  PERFORMANCE SETTINGS:")
            print(f"   • Max Workers: {perf_settings.get('max_workers', 'Unknown')}")
            print(f"   • Batch Size: {perf_settings.get('batch_size', 'Unknown')}")
            print(f"   • Image Quality: {perf_settings.get('image_quality', 'Unknown')}%")
            print(f"   • PDF DPI: {perf_settings.get('pdf_dpi', 'Unknown')}")
            
            # Display recommendations
            recommendations = self.rag_pipeline.get_performance_recommendations()
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
        except Exception as e:
            print(f"❌ Error retrieving performance stats: {e}")
    
    def display_vector_store_details(self) -> None:
        """Display detailed vector store information."""
        print("\n🗄️  VECTOR STORE DETAILS")
        print("-" * 40)
        
        try:
            vector_stats = self.rag_pipeline.vector_store.get_collection_statistics()
            
            print(f"📊 COLLECTION OVERVIEW:")
            print(f"   • Total Documents: {vector_stats.get('total_documents', 0)}")
            print(f"   • Embedding Model: {vector_stats.get('embedding_model', 'Unknown')}")
            print(f"   • Storage Directory: {vector_stats.get('persist_directory', 'Unknown')}")
            
            collections = vector_stats.get('collections', {})
            if collections:
                print(f"\n📂 COLLECTION BREAKDOWN:")
                for coll_type, coll_info in collections.items():
                    count = coll_info.get('document_count', 0)
                    name = coll_info.get('collection_name', 'Unknown')
                    print(f"   • {coll_type.title()}: {count} documents ({name})")
            
            # Show document sources
            sources = self.rag_pipeline.vector_store.get_document_sources()
            print(f"\n📁 INGESTED DOCUMENTS ({len(sources)} files):")
            for source in sources:
                filename = Path(source).name
                print(f"   • {filename}")
            
        except Exception as e:
            print(f"❌ Error retrieving vector store details: {e}")
    
    def display_cache_information(self) -> None:
        """Display cache information and statistics."""
        print("\n💾 CACHE INFORMATION")
        print("-" * 40)
        
        try:
            # Get extraction cache status
            cached_extractions = self.rag_pipeline.document_processor.get_all_saved_extractions()
            
            if not cached_extractions:
                print("📭 No cached extractions found.")
                return
            
            total_text = sum(ext['content_stats']['text_items'] for ext in cached_extractions)
            total_tables = sum(ext['content_stats']['table_items'] for ext in cached_extractions)
            total_visuals = sum(ext['content_stats']['visual_items'] for ext in cached_extractions)
            
            print(f"📊 CACHE SUMMARY:")
            print(f"   • Cached Files: {len(cached_extractions)}")
            print(f"   • Total Text Items: {total_text}")
            print(f"   • Total Table Items: {total_tables}")
            print(f"   • Total Visual Items: {total_visuals}")
            
            # Show recent extractions
            recent_extractions = sorted(
                cached_extractions, 
                key=lambda x: x['extraction_date'], 
                reverse=True
            )[:5]
            
            print(f"\n📅 RECENT EXTRACTIONS:")
            for ext in recent_extractions:
                source_name = Path(ext['source_file']).name
                stats = ext['content_stats']
                date = ext['extraction_date'][:10]
                print(f"   • {source_name} ({date})")
                print(f"     Text: {stats['text_items']}, Tables: {stats['table_items']}, Visuals: {stats['visual_items']}")
            
        except Exception as e:
            print(f"❌ Error retrieving cache information: {e}")
    
    # =============================================================================
    # SETTINGS AND PERFORMANCE MANAGEMENT
    # =============================================================================
    
    def handle_settings_and_performance(self) -> None:
        """Handle settings and performance management."""
        print("\n" + "="*50)
        print("⚙️  SETTINGS & PERFORMANCE")
        print("="*50)
        print("1. 📊 View Performance Configuration")
        print("2. 💡 Get Performance Recommendations")
        print("3. 🧹 System Maintenance")
        print("4. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            self.view_performance_configuration()
        elif choice == "2":
            self.show_performance_recommendations()
        elif choice == "3":
            self.system_maintenance()
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice. Please try again.")
    
    def view_performance_configuration(self) -> None:
        """Display current performance configuration."""
        print("\n📊 PERFORMANCE CONFIGURATION")
        print("-" * 40)
        
        try:
            performance_stats = self.rag_pipeline.document_processor.get_performance_statistics()
            
            opt_status = performance_stats.get("optimization_status", {})
            perf_settings = performance_stats.get("performance_settings", {})
            
            print("🚀 OPTIMIZATION FEATURES:")
            print(f"   • Batch Processing: {'✅ ON' if opt_status.get('batch_processing_enabled') else '❌ OFF'}")
            print(f"   • Image Compression: {'✅ ON' if opt_status.get('image_compression_enabled') else '❌ OFF'}")
            print(f"   • Smart Filtering: {'✅ ON' if opt_status.get('smart_filtering_enabled') else '❌ OFF'}")
            print(f"   • Parallel Processing: {'✅ ON' if opt_status.get('parallel_processing_enabled') else '❌ OFF'}")
            print(f"   • Duplicate Detection: {'✅ ON' if opt_status.get('duplicate_detection_enabled') else '❌ OFF'}")
            print(f"   • Extraction Caching: {'✅ ON' if opt_status.get('extraction_caching_enabled') else '❌ OFF'}")
            
            print(f"\n⚙️  PERFORMANCE PARAMETERS:")
            print(f"   • Max Workers: {perf_settings.get('max_workers', 'Unknown')}")
            print(f"   • Image Quality: {perf_settings.get('image_quality', 'Unknown')}%")
            print(f"   • PDF DPI: {perf_settings.get('pdf_dpi', 'Unknown')}")
            print(f"   • Min Pages for Parallel: {perf_settings.get('min_pages_for_parallel', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error retrieving performance configuration: {e}")
    
    def show_performance_recommendations(self) -> None:
        """Show actionable performance recommendations."""
        print("\n💡 PERFORMANCE RECOMMENDATIONS")
        print("-" * 40)
        
        try:
            recommendations = self.rag_pipeline.get_performance_recommendations()
            
            if not recommendations:
                print("✅ Your system is already well-optimized!")
                return
            
            print("📋 Recommendations to improve performance:")
            for i, recommendation in enumerate(recommendations, 1):
                print(f"   {i}. {recommendation}")
            
            print(f"\n💡 TIP: These settings can be adjusted in the config.py file.")
            
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
    
    def system_maintenance(self) -> None:
        """Handle system maintenance operations."""
        print("\n🧹 SYSTEM MAINTENANCE")
        print("-" * 40)
        print("1. 🗑️  Clear Old Cache Files")
        print("2. 🔄 Reset Duplicate Detection")
        print("3. 📊 Analyze Storage Usage")
        print("4. 🔙 Back")
        
        choice = input("\n🎯 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            self.clear_old_cache()
        elif choice == "2":
            self.reset_duplicate_detection()
        elif choice == "3":
            self.analyze_storage_usage()
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice. Please try again.")
    
    def reset_duplicate_detection(self) -> None:
        """Reset the duplicate detection cache."""
        print("\n🔄 RESET DUPLICATE DETECTION")
        print("-" * 40)
        
        confirm = input("⚠️  This will reset duplicate detection cache. Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            try:
                self.rag_pipeline.document_processor.reset_duplicate_detection()
                print("✅ Duplicate detection cache reset successfully!")
            except Exception as e:
                print(f"❌ Error resetting duplicate detection: {e}")
        else:
            print("Operation cancelled.")
    
    def analyze_storage_usage(self) -> None:
        """Analyze and display storage usage information."""
        print("\n📊 STORAGE USAGE ANALYSIS")
        print("-" * 40)
        
        try:
            # Basic storage analysis
            current_dir = Path.cwd()
            
            # Check vector store directory
            persist_dir = Path("chroma_db")
            if persist_dir.exists():
                vector_size = sum(f.stat().st_size for f in persist_dir.rglob('*') if f.is_file())
                vector_size_mb = vector_size / (1024 * 1024)
                print(f"🗄️  Vector Store: {vector_size_mb:.2f} MB")
            
            # Check extraction cache directory
            cache_dir = Path("document_extractions")
            if cache_dir.exists():
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)
                cache_files = len(list(cache_dir.rglob('*.json')))
                print(f"💾 Extraction Cache: {cache_size_mb:.2f} MB ({cache_files} files)")
            
            # Check logs directory
            logs_dir = self.log_directory
            if logs_dir.exists():
                logs_size = sum(f.stat().st_size for f in logs_dir.rglob('*') if f.is_file())
                logs_size_mb = logs_size / (1024 * 1024)
                log_files = len(list(logs_dir.rglob('*.txt')))
                print(f"📝 Logs: {logs_size_mb:.2f} MB ({log_files} files)")
            
            total_size_mb = vector_size_mb + cache_size_mb + logs_size_mb
            print(f"\n📊 Total Storage Used: {total_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error analyzing storage usage: {e}")
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _save_error_log(self, file_path: str, error_message: str) -> None:
        """Save error information to log file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ERROR_{Path(file_path).stem}_{timestamp}.txt"
            log_path = self.log_directory / filename
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"ERROR LOG\n")
                f.write(f"{'='*50}\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_message}\n")
            
            print(f"📝 Error log saved: {log_path}")
            
        except Exception as e:
            print(f"❌ Failed to save error log: {e}")
    
    def handle_exit(self) -> None:
        """Handle graceful exit from the application."""
        print("\n👋 Thank you for using Multi-Modal RAG Pipeline!")
        print("📊 Session Summary:")
        
        try:
            # Display quick session summary
            vector_stats = self.rag_pipeline.vector_store.get_collection_statistics()
            total_docs = vector_stats.get('total_documents', 0)
            print(f"   • Documents in knowledge base: {total_docs}")
            print(f"   • System ready for questions: {'Yes' if total_docs > 0 else 'No'}")
        except:
            pass
        
        print("🚀 Pipeline ready for next session!")


def main():
    """Main entry point for the RAG Pipeline CLI."""
    try:
        interface = RAGInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
