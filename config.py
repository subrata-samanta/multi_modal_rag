import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Configuration settings for the Multi-Modal RAG System
    
    This class contains all configuration parameters for document processing,
    vector storage, performance optimization, and AI model settings.
    """
    
    # =============================================================================
    # AUTHENTICATION & CREDENTIALS
    # =============================================================================
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "key.json")
    
    # =============================================================================
    # VECTOR DATABASE SETTINGS
    # =============================================================================
    # Text chunking configuration for vector storage
    CHUNK_SIZE = 1000                    # Maximum characters per text chunk
    CHUNK_OVERLAP = 200                  # Character overlap between chunks
    
    # Vector database configuration
    COLLECTION_NAME = "multimodal_rag"   # Name of the vector collection
    PERSIST_DIRECTORY = "./chroma_db"    # Directory to store vector database
    IMAGES_STORAGE_PATH = "./stored_images"  # Path to store original images
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION SETTINGS
    # =============================================================================
    
    # --- Parallel Processing Configuration ---
    MAX_WORKERS = 6                      # Number of parallel workers for processing pages
    BATCH_SIZE = 5                       # Number of pages to process together in each batch
    ENABLE_MULTIPROCESSING = True        # Enable/disable parallel processing
    
    # --- Speed & Efficiency Optimizations ---
    ENABLE_BATCH_PROCESSING = True       # Process multiple content types in single API call (3x faster)
    ENABLE_IMAGE_COMPRESSION = True      # Compress images before sending to API (2x faster)
    ENABLE_SMART_FILTERING = True        # Skip pages with minimal content
    ENABLE_DUPLICATE_DETECTION = True    # Skip duplicate/similar pages
    
    # --- Image Processing Configuration ---
    IMAGE_QUALITY = 85                   # JPEG compression quality (70-95 range)
    IMAGE_MAX_SIZE = (1200, 1600)        # Maximum image dimensions (width, height)
    
    # --- Smart Filtering Thresholds ---
    MIN_TEXT_LENGTH = 50                 # Minimum text length to consider page worth processing
    SIMILARITY_THRESHOLD = 0.85          # Skip pages too similar to previous ones (0.0-1.0)
    
    # =============================================================================
    # AI MODEL PROMPTS
    # =============================================================================
    
    # --- Individual Content Type Prompts ---
    TEXT_EXTRACTION_PROMPT = """
    Extract all readable text content from this image.
    Focus on paragraphs, headings, bullet points, and any textual information.
    Do NOT extract tables or visual elements here - only pure text content.
    Preserve the structure and formatting as much as possible.
    If there is no text content, respond with "No text content found."
    """
    
    TABLE_EXTRACTION_PROMPT = """
    Analyze this image for any tables or tabular data.

    If tables are present, for EACH table provide:

    1. **Table Description**: A brief description of what the table represents
    2. **Table Data**: Extract the table in Markdown format with proper alignment
       - Use | for column separators
       - Use |---|---| for header separators
       - Preserve all rows and columns accurately


    Format your response as:

    ### Table 1: [Brief Title]
    **Description**: [What this table shows]

    **Table Data**:
    | Column 1 | Column 2 | Column 3 |
    |----------|----------|----------|
    | Data 1   | Data 2   | Data 3   |
    | Data 4   | Data 5   | Data 6   |


    If there are multiple tables, repeat this format for each table.
    If NO tables are present, respond with "No tables found."
    """
    
    VISUAL_ANALYSIS_PROMPT = """
    Analyze this image for any visual elements such as charts, graphs, diagrams, images, or illustrations.
    Do NOT analyze text or tables - focus only on visual/graphical content.

    If visual elements are present, for EACH visual element provide:

    1. **Visual Type**: Identify the type (e.g., Bar Chart, Pie Chart, Line Graph, Diagram, Flowchart, Image, Icon, etc.)
    2. **Visual Description**: Describe what the visual shows in detail
    3. **Data Summary**: If it's a chart/graph, extract the data in a tabular format:

    | Category/Label | Value/Metric | Additional Info |
    |----------------|--------------|-----------------|
    | Item 1         | Value 1      | Detail 1        |
    | Item 2         | Value 2      | Detail 2        |

    4. **Key Insights**: List 2-3 key insights, trends, or important observations from the visual

    Format your response as:

    ### Visual 1: [Type] - [Brief Title]
    **Description**: [Detailed description of what this visual represents]

    **Data Summary**:
    | Category | Value | Notes |
    |----------|-------|-------|
    | Data 1   | Val 1 | Note 1|
    | Data 2   | Val 2 | Note 2|

    **Key Insights**:
    - Insight 1: [Observation or trend]
    - Insight 2: [Pattern or comparison]
    - Insight 3: [Notable finding]

    **Additional Context**: [Any other relevant information about colors, legends, annotations, etc.]

    If there are multiple visual elements, repeat this format for each one.
    If NO visual elements are present, respond with "No visual elements found."
    """
    
    # --- Optimized Batch Processing Prompt (3x faster) ---
    BATCH_EXTRACTION_PROMPT = """
    Analyze this image and extract ALL content types. Organize your response in the following sections:

    ## TEXT CONTENT
    Extract all readable text content (paragraphs, headings, bullet points).
    If no text: "No text content found."

    ## TABLE CONTENT  
    If tables are present, for EACH table provide:
    ### Table X: [Brief Title]
    **Description**: [What this table shows]
    **Table Data**:
    | Column 1 | Column 2 | Column 3 |
    |----------|----------|----------|
    | Data 1   | Data 2   | Data 3   |
    
    If no tables: "No tables found."

    ## VISUAL CONTENT
    For charts, graphs, diagrams, or visual elements:
    ### Visual X: [Type] - [Brief Title]
    **Description**: [Detailed description]
    **Data Summary**:
    | Category | Value | Notes |
    |----------|-------|-------|
    | Item 1   | Val 1 | Note 1|
    
    **Key Insights**:
    - Insight 1: [Observation]
    - Insight 2: [Pattern]
    
    If no visuals: "No visual elements found."
    
    Be concise but comprehensive. Skip empty sections.
    """
    
    # =============================================================================
    # SUPPORTED FILE FORMATS
    # =============================================================================
    SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.pptx', '.eml', '.xls', '.xlsx']
    SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # =============================================================================
    # DIRECTORY STRUCTURE
    # =============================================================================
    EXTRACTION_CACHE_DIR = "document_extractions"
    EXTRACTION_IMAGES_DIR = "document_extractions/images"
    EXTRACTION_CONTENT_DIR = "document_extractions/content"
    
    # =============================================================================
    # PERFORMANCE MONITORING
    # =============================================================================
    ENABLE_PERFORMANCE_LOGGING = True   # Log performance metrics
    ENABLE_PROGRESS_DISPLAY = True      # Show progress during processing
