import os
from rag_pipeline import MultiModalRAGPipeline
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def save_extraction_log(file_path, content, output_folder="extraction_logs", is_error=False):
    """Save extracted content to a text file for review"""
    # Create output folder if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sanitized filename
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_prefix = "ERROR_" if is_error else ""
    output_filename = f"{status_prefix}{original_filename}_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"EXTRACTION LOG {'[ERROR]' if is_error else '[SUCCESS]'}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Source File: {file_path}\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(content)
    
    return output_path

def display_and_log_page_content(page_num, text_content, table_content, visual_content, log_file, lock=None):
    """Display and log extracted content for a single page"""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"PAGE {page_num} EXTRACTION RESULTS")
    output.append(f"{'='*80}")
    
    # Prepare text content
    if text_content and text_content.strip() and "no text content found" not in text_content.lower():
        output.append(f"\n--- TEXT CONTENT ---")
        output.append(text_content[:500] + "..." if len(text_content) > 500 else text_content)
        log_text = f"\n{'='*80}\nPAGE {page_num} - TEXT CONTENT\n{'='*80}\n{text_content}\n\n"
    else:
        output.append(f"\n--- TEXT CONTENT ---")
        output.append("No text content found on this page.")
        log_text = f"\n{'='*80}\nPAGE {page_num} - TEXT CONTENT\n{'='*80}\nNo text content found on this page.\n\n"
    
    # Prepare table content
    if table_content and table_content.strip() and "no tables found" not in table_content.lower():
        output.append(f"\n--- TABLE CONTENT ---")
        output.append(table_content[:500] + "..." if len(table_content) > 500 else table_content)
        log_table = f"{'='*80}\nPAGE {page_num} - TABLE CONTENT\n{'='*80}\n{table_content}\n\n"
    else:
        output.append(f"\n--- TABLE CONTENT ---")
        output.append("No tables found on this page.")
        log_table = f"{'='*80}\nPAGE {page_num} - TABLE CONTENT\n{'='*80}\nNo tables found on this page.\n\n"
    
    # Prepare visual content
    if visual_content and visual_content.strip() and "no visual elements found" not in visual_content.lower():
        output.append(f"\n--- VISUAL CONTENT ---")
        output.append(visual_content[:500] + "..." if len(visual_content) > 500 else visual_content)
        log_visual = f"{'='*80}\nPAGE {page_num} - VISUAL CONTENT\n{'='*80}\n{visual_content}\n\n"
    else:
        output.append(f"\n--- VISUAL CONTENT ---")
        output.append("No visual elements found on this page.")
        log_visual = f"{'='*80}\nPAGE {page_num} - VISUAL CONTENT\n{'='*80}\nNo visual elements found on this page.\n\n"
    
    output.append(f"{'='*80}\n")
    
    # Thread-safe printing and logging
    if lock:
        # For parallel processing, use tqdm.write for thread-safe output
        from tqdm import tqdm
        tqdm.write('\n'.join(output))
        if log_file:
            log_file.write(log_text)
            log_file.write(log_table)
            log_file.write(log_visual)
    else:
        print('\n'.join(output))
        if log_file:
            log_file.write(log_text)
            log_file.write(log_table)
            log_file.write(log_visual)

def ingest_mode(rag):
    """Handle document/folder ingestion"""
    print("\n=== INGESTION MODE ===")
    print("1. Ingest Single Document")
    print("2. Ingest Folder")
    print("3. Load from Saved Extractions")
    print("4. View Extraction Cache Status")
    print("5. Clear Old Extractions")
    print("6. Force Reprocess Document")
    print("7. Back to Main Menu")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == "1":
        file_path = input("Enter the path to your document file (PDF, PPTX, EML, XLS, XLSX): ").strip()
        
        if not os.path.exists(file_path):
            print("File not found. Please check the path.")
            return
        
        if not file_path.lower().endswith(('.pdf', '.pptx', '.eml', '.xls', '.xlsx')):
            print("Unsupported file type. Please use PDF, PPTX, EML, XLS, or XLSX files.")
            return
        
        # Check if document has cached extraction
        if rag.document_processor.has_saved_extraction(file_path):
            use_cache = input("Found cached extraction. Use cache? (y/n, default: y): ").strip().lower()
            force_reprocess = use_cache == 'n'
        else:
            force_reprocess = False
        
        try:
            print(f"\nProcessing document: {file_path}")
            
            result = rag.ingest_document(file_path, force_reprocess)
            
            print("\n✓ Document ingested successfully!")
            
            # Show extraction stats
            text_count = len(result.get('text', []))
            table_count = len(result.get('tables', []))
            visual_count = len(result.get('visuals', []))
            print(f"  - Text items: {text_count}")
            print(f"  - Table items: {table_count}")
            print(f"  - Visual items: {visual_count}")
            
        except Exception as e:
            print(f"✗ Error processing document: {e}")
    
    elif choice == "2":
        folder_path = input("Enter the path to the folder containing document files (PDF, PPTX, EML, XLS, XLSX): ").strip()
        
        if not os.path.exists(folder_path):
            print("Folder not found. Please check the path.")
            return
        
        if not os.path.isdir(folder_path):
            print("The provided path is not a folder.")
            return
        
        # Ask for force reprocessing
        force_reprocess = input("Force reprocess all files (ignore cache)? (y/n, default: n): ").strip().lower() == 'y'
        
        try:
            results = rag.ingest_folder_with_caching(folder_path, force_reprocess)
            
            print(f"\n✓ Folder ingestion complete!")
            print(f"Processed {len(results)} files")
            
        except Exception as e:
            print(f"✗ Error processing folder: {e}")
    
    elif choice == "3":
        # Load from saved extractions
        extraction_dir = input("Enter extraction directory path (default: document_extractions/content): ").strip()
        if not extraction_dir:
            extraction_dir = "document_extractions/content"
        
        try:
            rag.ingest_from_saved_extractions(extraction_dir)
        except Exception as e:
            print(f"✗ Error loading saved extractions: {e}")
    
    elif choice == "4":
        # View extraction cache status
        try:
            rag.get_extraction_status()
        except Exception as e:
            print(f"✗ Error getting extraction status: {e}")
    
    elif choice == "5":
        # Clear old extractions
        days = input("Enter number of days to keep recent extractions (default: 30): ").strip()
        try:
            keep_days = int(days) if days else 30
            rag.clear_old_extractions(keep_days)
        except ValueError:
            print("Invalid number of days. Using default (30 days).")
            rag.clear_old_extractions(30)
        except Exception as e:
            print(f"✗ Error clearing extractions: {e}")
    
    elif choice == "6":
        # Force reprocess document
        file_path = input("Enter the path to the document to force reprocess: ").strip()
        
        if not os.path.exists(file_path):
            print("File not found. Please check the path.")
            return
        
        if not file_path.lower().endswith(('.pdf', '.pptx', '.eml', '.xls', '.xlsx')):
            print("Unsupported file type. Please use PDF, PPTX, EML, XLS, or XLSX files.")
            return
        
        try:
            result = rag.force_reprocess_document(file_path)
            print("\n✓ Document reprocessed and ingested successfully!")
            
            # Show extraction stats
            text_count = len(result.get('text', []))
            table_count = len(result.get('tables', []))
            visual_count = len(result.get('visuals', []))
            print(f"  - Text items: {text_count}")
            print(f"  - Table items: {table_count}")
            print(f"  - Visual items: {visual_count}")
            
        except Exception as e:
            print(f"✗ Error reprocessing document: {e}")
    
    elif choice == "7":
        return
    else:
        print("Invalid choice. Please try again.")

def inference_mode(rag):
    """Handle question answering with mode selection"""
    print("\n=== INFERENCE MODE ===")
    print("Select inference type:")
    print("1. Auto (intelligent selection)")
    print("2. Text-only (faster)")
    print("3. Multimodal (comprehensive)")
    print("4. Back to Main Menu")
    
    mode_choice = input("\nEnter your choice (1-4): ").strip()
    
    if mode_choice == "4":
        return
    
    mode_map = {"1": "auto", "2": "text", "3": "multimodal"}
    inference_type = mode_map.get(mode_choice)
    
    if not inference_type:
        print("Invalid choice.")
        return
    
    print(f"\nInference mode: {inference_type.upper()}")
    question = input("Enter your question: ").strip()
    
    if not question:
        print("Please enter a valid question.")
        return
    
    print("\nProcessing your question...")
    try:
        result = rag.answer_question(question, mode=inference_type)
        
        print(f"\n=== ANSWER ===")
        print(result["answer"])
        
        print(f"\n=== SOURCES ===")
        for source in result["sources"]:
            print(f"- {source['source']} (Page {source['page']}, Type: {source['type']})")
        
        print(f"\n=== RETRIEVAL INFO ===")
        print(f"Documents retrieved: {result['num_documents_retrieved']}")
        
    except Exception as e:
        print(f"Error answering question: {e}")

def main():
    # Initialize the RAG pipeline
    rag = MultiModalRAGPipeline()
    
    print("=== Multimodal RAG Pipeline ===")
    print("This pipeline can process PDF, PPTX, EML, XLS, and XLSX files using Google Gemini Vision")
    print("It extracts text, tables, and visual information for comprehensive Q&A")
    print()
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Ingestion (Add documents)")
        print("2. Inference (Ask questions)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            ingest_mode(rag)
        
        elif choice == "2":
            inference_mode(rag)
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
