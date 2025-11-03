import os
import base64
from typing import List, Dict, Any
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from pptx import Presentation
from pptx.util import Inches
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import email
from email import policy
from email.parser import BytesParser
import openpyxl
import xlrd
from openpyxl.drawing.image import Image as OpenpyxlImage
import zipfile
import tempfile
import json
import hashlib
from datetime import datetime

class DocumentProcessor:
    def __init__(self):
        self._setup_authentication()
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-pro"
        )
        self._thread_local = threading.local()
        
        # Create directories for saving extractions
        self.extractions_dir = "document_extractions"
        self.images_dir = os.path.join(self.extractions_dir, "images")
        self.content_dir = os.path.join(self.extractions_dir, "content")
        
        os.makedirs(self.extractions_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)
        
    def _setup_authentication(self):
        """Setup Google authentication using service account key file"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
        except Exception as e:
            raise Exception(f"Failed to setup Google authentication: {e}")
    
    def _get_llm(self):
        """Get thread-local LLM instance"""
        if not hasattr(self._thread_local, 'llm'):
            self._thread_local.llm = ChatVertexAI(
                model_name="gemini-2.5-pro"
            )
        return self._thread_local.llm
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for caching"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_extraction_file_path(self, file_path: str) -> str:
        """Get the path where extraction results should be saved"""
        file_hash = self._get_file_hash(file_path)
        filename = os.path.basename(file_path)
        extraction_filename = f"{filename}_{file_hash}.json"
        return os.path.join(self.content_dir, extraction_filename)
    
    def _save_image(self, image: Image.Image, file_path: str, page_num: int, content_type: str) -> str:
        """Save image to disk and return the saved path"""
        try:
            file_hash = self._get_file_hash(file_path)
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            image_filename = f"{base_name}_{file_hash}_page{page_num}_{content_type}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            
            image.save(image_path, "PNG")
            return image_path
        except Exception as e:
            print(f"Error saving image: {e}")
            return ""
    
    def _save_extraction_results(self, file_path: str, processed_content: Dict[str, List[Dict[str, Any]]]):
        """Save extraction results to JSON file"""
        try:
            extraction_path = self._get_extraction_file_path(file_path)
            
            # Add metadata
            extraction_data = {
                "source_file": file_path,
                "file_hash": self._get_file_hash(file_path),
                "extraction_date": datetime.now().isoformat(),
                "content": processed_content
            }
            
            with open(extraction_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved extraction results to: {extraction_path}")
            return extraction_path
            
        except Exception as e:
            print(f"Error saving extraction results: {e}")
            return None
    
    def _load_saved_extraction(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load previously saved extraction results"""
        try:
            extraction_path = self._get_extraction_file_path(file_path)
            
            if not os.path.exists(extraction_path):
                return None
            
            # Check if source file has been modified since extraction
            current_hash = self._get_file_hash(file_path)
            
            with open(extraction_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            saved_hash = extraction_data.get("file_hash", "")
            
            if current_hash != saved_hash:
                print(f"File has been modified since extraction. Re-processing required.")
                return None
            
            print(f"✓ Loaded cached extraction from: {extraction_path}")
            return extraction_data["content"]
            
        except Exception as e:
            print(f"Error loading saved extraction: {e}")
            return None
    
    def has_saved_extraction(self, file_path: str) -> bool:
        """Check if file has saved extraction that is still valid"""
        saved_content = self._load_saved_extraction(file_path)
        return saved_content is not None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images using PyMuPDF (no poppler dependency)"""
        try:
            images = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Render page to image with high resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def convert_pptx_to_images(self, pptx_path: str) -> List[Image.Image]:
        """Convert PPTX slides to images using python-pptx"""
        try:
            images = []
            prs = Presentation(pptx_path)
            
            # Create temporary directory for slide images
            temp_dir = "temp_slides"
            os.makedirs(temp_dir, exist_ok=True)
            
            for slide_num, slide in enumerate(prs.slides):
                # Export slide as image
                slide_image_path = os.path.join(temp_dir, f"slide_{slide_num}.png")
                
                # Get slide dimensions
                slide_width = prs.slide_width
                slide_height = prs.slide_height
                
                # Create blank image with slide dimensions
                img = Image.new('RGB', (int(slide_width / 9525), int(slide_height / 9525)), 'white')
                
                # Note: python-pptx doesn't directly support image export
                # For production, consider using win32com (Windows) or LibreOffice (cross-platform)
                # This is a placeholder - slides will be processed as text extraction
                images.append(img)
            
            # Clean up temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return images
        except Exception as e:
            print(f"Error converting PPTX: {e}")
            return []
    
    def process_eml_file(self, eml_path: str) -> Dict[str, Any]:
        """Process EML (email) file and extract content"""
        try:
            with open(eml_path, 'rb') as f:
                raw_email = f.read()
            
            # Parse the email
            msg = BytesParser(policy=policy.default).parsebytes(raw_email)
            
            # Extract email metadata and content
            email_content = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'body_text': '',
                'body_html': '',
                'attachments': []
            }
            
            # Extract body content
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition', ''))
                    
                    if content_type == 'text/plain' and 'attachment' not in content_disposition:
                        email_content['body_text'] = part.get_content()
                    elif content_type == 'text/html' and 'attachment' not in content_disposition:
                        email_content['body_html'] = part.get_content()
                    elif 'attachment' in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            email_content['attachments'].append({
                                'filename': filename,
                                'content_type': content_type,
                                'size': len(part.get_payload(decode=True) or b'')
                            })
            else:
                # Single part message
                email_content['body_text'] = msg.get_content()
            
            return email_content
            
        except Exception as e:
            print(f"Error processing EML file: {e}")
            return {}
    
    def extract_images_from_xlsx(self, xlsx_path: str) -> List[Image.Image]:
        """Extract embedded images from XLSX file"""
        images = []
        try:
            # Open the workbook
            workbook = openpyxl.load_workbook(xlsx_path)
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                
                # Check if sheet has images
                if hasattr(sheet, '_images') and sheet._images:
                    for img in sheet._images:
                        try:
                            # Get image data
                            image_data = img._data()
                            pil_image = Image.open(BytesIO(image_data))
                            images.append(pil_image)
                        except Exception as e:
                            print(f"Error extracting image from sheet {sheet_name}: {e}")
            
            workbook.close()
            
        except Exception as e:
            print(f"Error extracting images from XLSX: {e}")
        
        return images
    
    def convert_excel_to_images(self, excel_path: str) -> List[Image.Image]:
        """Convert Excel sheets to images for processing"""
        try:
            images = []
            file_ext = os.path.splitext(excel_path)[1].lower()
            
            if file_ext == '.xlsx':
                # Use openpyxl for .xlsx files
                workbook = openpyxl.load_workbook(excel_path, data_only=True)
                
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    
                    # Create a visual representation of the sheet
                    # This is a simplified approach - for better visualization,
                    # consider using libraries like xlwings or converting via LibreOffice
                    sheet_image = self._create_sheet_visualization(sheet, sheet_name)
                    if sheet_image:
                        images.append(sheet_image)
                
                # Also extract any embedded images
                embedded_images = self.extract_images_from_xlsx(excel_path)
                images.extend(embedded_images)
                
                workbook.close()
                
            elif file_ext == '.xls':
                # Use xlrd for .xls files
                workbook = xlrd.open_workbook(excel_path)
                
                for sheet_index in range(workbook.nsheets):
                    sheet = workbook.sheet_by_index(sheet_index)
                    sheet_name = workbook.sheet_names()[sheet_index]
                    
                    # Create a visual representation of the sheet
                    sheet_image = self._create_xls_sheet_visualization(sheet, sheet_name)
                    if sheet_image:
                        images.append(sheet_image)
            
            return images
            
        except Exception as e:
            print(f"Error converting Excel file: {e}")
            return []
    
    def _create_sheet_visualization(self, sheet, sheet_name: str) -> Image.Image:
        """Create a visual representation of an Excel sheet (xlsx)"""
        try:
            # Get sheet dimensions
            max_row = sheet.max_row
            max_col = sheet.max_column
            
            if max_row == 1 and max_col == 1 and not sheet.cell(1, 1).value:
                return None  # Empty sheet
            
            # Create a simple text-based visualization
            # For better results, consider using xlwings or similar libraries
            cell_width = 120
            cell_height = 30
            
            img_width = min(max_col * cell_width, 2000)  # Limit width
            img_height = min(max_row * cell_height, 2000)  # Limit height
            
            # Create blank image
            img = Image.new('RGB', (img_width, img_height), 'white')
            
            # This is a placeholder implementation
            # In production, you might want to use more sophisticated Excel-to-image conversion
            
            return img
            
        except Exception as e:
            print(f"Error creating sheet visualization: {e}")
            return None
    
    def _create_xls_sheet_visualization(self, sheet, sheet_name: str) -> Image.Image:
        """Create a visual representation of an Excel sheet (xls)"""
        try:
            # Get sheet dimensions
            nrows = sheet.nrows
            ncols = sheet.ncols
            
            if nrows == 0 or ncols == 0:
                return None  # Empty sheet
            
            # Create a simple text-based visualization
            cell_width = 120
            cell_height = 30
            
            img_width = min(ncols * cell_width, 2000)  # Limit width
            img_height = min(nrows * cell_height, 2000)  # Limit height
            
            # Create blank image
            img = Image.new('RGB', (img_width, img_height), 'white')
            
            # This is a placeholder implementation
            # In production, you might want to use more sophisticated Excel-to-image conversion
            
            return img
            
        except Exception as e:
            print(f"Error creating XLS sheet visualization: {e}")
            return None
    
    def extract_content_from_image(self, image: Image.Image, content_type: str) -> str:
        """Extract specific content type from image using ChatVertexAI"""
        try:
            if content_type == "text":
                prompt = Config.TEXT_EXTRACTION_PROMPT
            elif content_type == "table":
                prompt = Config.TABLE_EXTRACTION_PROMPT
            elif content_type == "visual":
                prompt = Config.VISUAL_ANALYSIS_PROMPT
            else:
                raise ValueError(f"Unknown content type: {content_type}")
            
            # Convert image to bytes
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            
            # Create message with image using the correct format for Vertex AI
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                        }
                    }
                ]
            )
            
            # Use thread-local LLM instance
            llm = self._get_llm()
            response = llm.invoke([message])
            
            # Handle different response types
            if hasattr(response, 'content'):
                # If response has content attribute, use it
                result = response.content
            elif isinstance(response, dict):
                # If response is a dict, try to get content
                result = response.get('content', str(response))
            else:
                # Otherwise convert to string
                result = str(response)
            
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            print(f"Error extracting {content_type}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def process_eml_document(self, eml_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process EML file and extract content for RAG pipeline"""
        try:
            email_content = self.process_eml_file(eml_path)
            
            processed_content = {
                "text": [],
                "tables": [],
                "visuals": []
            }
            
            # Process email metadata and body as text
            email_text_parts = []
            
            if email_content.get('subject'):
                email_text_parts.append(f"Subject: {email_content['subject']}")
            
            if email_content.get('from'):
                email_text_parts.append(f"From: {email_content['from']}")
            
            if email_content.get('to'):
                email_text_parts.append(f"To: {email_content['to']}")
            
            if email_content.get('date'):
                email_text_parts.append(f"Date: {email_content['date']}")
            
            if email_content.get('body_text'):
                email_text_parts.append(f"Body:\n{email_content['body_text']}")
            elif email_content.get('body_html'):
                # Basic HTML stripping for HTML-only emails
                import re
                html_content = email_content['body_html']
                # Remove HTML tags
                clean_text = re.sub('<[^<]+?>', '', html_content)
                email_text_parts.append(f"Body:\n{clean_text}")
            
            if email_content.get('attachments'):
                attachment_info = []
                for att in email_content['attachments']:
                    attachment_info.append(f"- {att['filename']} ({att['content_type']}, {att['size']} bytes)")
                if attachment_info:
                    email_text_parts.append(f"Attachments:\n" + "\n".join(attachment_info))
            
            # Combine all email content
            full_email_text = "\n\n".join(email_text_parts)
            
            processed_content["text"].append({
                "content": full_email_text,
                "page": 1,
                "source": eml_path,
                "type": "email"
            })
            
            return processed_content
            
        except Exception as e:
            print(f"Error processing EML document: {e}")
            return {"text": [], "tables": [], "visuals": []}
    
    def process_excel_document(self, excel_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process Excel file and extract content for RAG pipeline"""
        try:
            file_ext = os.path.splitext(excel_path)[1].lower()
            
            processed_content = {
                "text": [],
                "tables": [],
                "visuals": []
            }
            
            if file_ext == '.xlsx':
                workbook = openpyxl.load_workbook(excel_path, data_only=True)
                
                for sheet_index, sheet_name in enumerate(workbook.sheetnames):
                    sheet = workbook[sheet_name]
                    
                    # Extract sheet data as tables
                    sheet_data = self._extract_xlsx_sheet_data(sheet, sheet_name)
                    if sheet_data:
                        processed_content["tables"].append({
                            "content": sheet_data,
                            "page": sheet_index + 1,
                            "source": excel_path,
                            "type": "excel_sheet",
                            "sheet_name": sheet_name
                        })
                
                # Extract embedded images
                embedded_images = self.extract_images_from_xlsx(excel_path)
                for img_index, image in enumerate(embedded_images):
                    visual_content = self.extract_content_from_image(image, "visual")
                    if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                        processed_content["visuals"].append({
                            "content": visual_content,
                            "page": f"embedded_image_{img_index + 1}",
                            "source": excel_path,
                            "type": "embedded_image"
                        })
                
                workbook.close()
                
            elif file_ext == '.xls':
                workbook = xlrd.open_workbook(excel_path)
                
                for sheet_index in range(workbook.nsheets):
                    sheet = workbook.sheet_by_index(sheet_index)
                    sheet_name = workbook.sheet_names()[sheet_index]
                    
                    # Extract sheet data as tables
                    sheet_data = self._extract_xls_sheet_data(sheet, sheet_name)
                    if sheet_data:
                        processed_content["tables"].append({
                            "content": sheet_data,
                            "page": sheet_index + 1,
                            "source": excel_path,
                            "type": "excel_sheet",
                            "sheet_name": sheet_name
                        })
            
            return processed_content
            
        except Exception as e:
            print(f"Error processing Excel document: {e}")
            return {"text": [], "tables": [], "visuals": []}
    
    def _extract_xlsx_sheet_data(self, sheet, sheet_name: str) -> str:
        """Extract data from XLSX sheet and format as markdown table"""
        try:
            # Get sheet dimensions
            max_row = sheet.max_row
            max_col = sheet.max_column
            
            if max_row == 1 and max_col == 1 and not sheet.cell(1, 1).value:
                return ""  # Empty sheet
            
            # Extract data
            data = []
            for row in range(1, min(max_row + 1, 101)):  # Limit to first 100 rows
                row_data = []
                for col in range(1, min(max_col + 1, 21)):  # Limit to first 20 columns
                    cell_value = sheet.cell(row, col).value
                    if cell_value is None:
                        cell_value = ""
                    row_data.append(str(cell_value))
                data.append(row_data)
            
            if not data:
                return ""
            
            # Format as markdown table
            markdown_table = f"### Sheet: {sheet_name}\n\n"
            
            # Create header
            if len(data) > 0:
                headers = data[0]
                markdown_table += "| " + " | ".join(headers) + " |\n"
                markdown_table += "|" + "|".join(["---"] * len(headers)) + "|\n"
                
                # Add data rows
                for row in data[1:]:
                    if len(row) == len(headers):  # Ensure consistent column count
                        markdown_table += "| " + " | ".join(row) + " |\n"
            
            return markdown_table
            
        except Exception as e:
            print(f"Error extracting XLSX sheet data: {e}")
            return ""
    
    def _extract_xls_sheet_data(self, sheet, sheet_name: str) -> str:
        """Extract data from XLS sheet and format as markdown table"""
        try:
            nrows = sheet.nrows
            ncols = sheet.ncols
            
            if nrows == 0 or ncols == 0:
                return ""  # Empty sheet
            
            # Extract data
            data = []
            for row in range(min(nrows, 101)):  # Limit to first 100 rows
                row_data = []
                for col in range(min(ncols, 21)):  # Limit to first 20 columns
                    cell_value = sheet.cell_value(row, col)
                    if cell_value is None or cell_value == "":
                        cell_value = ""
                    row_data.append(str(cell_value))
                data.append(row_data)
            
            if not data:
                return ""
            
            # Format as markdown table
            markdown_table = f"### Sheet: {sheet_name}\n\n"
            
            # Create header
            if len(data) > 0:
                headers = data[0]
                markdown_table += "| " + " | ".join(headers) + " |\n"
                markdown_table += "|" + "|".join(["---"] * len(headers)) + "|\n"
                
                # Add data rows
                for row in data[1:]:
                    if len(row) == len(headers):  # Ensure consistent column count
                        markdown_table += "| " + " | ".join(row) + " |\n"
            
            return markdown_table
            
        except Exception as e:
            print(f"Error extracting XLS sheet data: {e}")
            return ""
        """Process a single page and extract all content types"""
        image = page_data['image']
        page_num = page_data['page_num']
        file_path = page_data['file_path']
        image_path = page_data.get('image_path', '')
        
        result = {
            'page_num': page_num,
            'text': None,
            'table': None,
            'visual': None
        }
        
        # Extract text content
        text_content = self.extract_content_from_image(image, "text")
        if text_content.strip() and "no text content found" not in text_content.lower():
            result['text'] = {
                "content": text_content,
                "page": page_num,
                "source": file_path,
                "type": "text",
                "image_path": image_path
            }
        
        # Extract table content
        table_content = self.extract_content_from_image(image, "table")
        if table_content.strip() and "no tables found" not in table_content.lower():
            result['table'] = {
                "content": table_content,
                "page": page_num,
                "source": file_path,
                "type": "table",
                "image_path": image_path
            }
        
        # Extract visual content
        visual_content = self.extract_content_from_image(image, "visual")
        if visual_content.strip() and "no visual elements found" not in visual_content.lower():
            result['visual'] = {
                "content": visual_content,
                "page": page_num,
                "source": file_path,
                "type": "visual",
                "image_path": image_path
            }
        
        return result
    
    def process_document_parallel(self, file_path: str, force_reprocess: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Process document with parallel processing and caching"""
        
        # Check for cached extraction first
        if not force_reprocess:
            saved_content = self._load_saved_extraction(file_path)
            if saved_content is not None:
                return saved_content
        
        print(f"Processing document: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle different file types
        if file_ext == '.eml':
            processed_content = self.process_eml_document(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            processed_content = self.process_excel_document(file_path)
        elif file_ext == '.pdf':
            images = self.convert_pdf_to_images(file_path)
            processed_content = self._process_images_parallel(file_path, images)
        elif file_ext == '.pptx':
            images = self.convert_pptx_to_images(file_path)
            processed_content = self._process_images_parallel(file_path, images)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Save extraction results
        self._save_extraction_results(file_path, processed_content)
        
        return processed_content
    
    def _process_images_parallel(self, file_path: str, images: List[Image.Image]) -> Dict[str, List[Dict[str, Any]]]:
        """Process images in parallel and save individual page images"""
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        if not Config.ENABLE_MULTIPROCESSING or len(images) < 2:
            # Fall back to sequential processing for small documents
            return self._process_images_sequential(file_path, images)
        
        # Save images first
        saved_images = []
        for page_num, image in enumerate(images):
            image_path = self._save_image(image, file_path, page_num + 1, "page")
            saved_images.append({
                'image': image,
                'image_path': image_path,
                'page_num': page_num + 1,
                'file_path': file_path
            })
        
        # Prepare page data for parallel processing
        page_data_list = saved_images
        
        # Process pages in parallel
        print(f"Processing {len(images)} pages with {Config.MAX_WORKERS} workers...")
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self.process_page, page_data): page_data['page_num']
                for page_data in page_data_list
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results[result['page_num']] = result
                    print(f"✓ Completed page {result['page_num']}/{len(images)}")
                except Exception as e:
                    print(f"✗ Error processing page {page_num}: {e}")
        
        # Sort results by page number and collect
        for page_num in sorted(results.keys()):
            result = results[page_num]
            
            if result['text']:
                processed_content["text"].append(result['text'])
            if result['table']:
                processed_content["tables"].append(result['table'])
            if result['visual']:
                processed_content["visuals"].append(result['visual'])
        
        return processed_content
    
    def _process_images_sequential(self, file_path: str, images: List[Image.Image]) -> Dict[str, List[Dict[str, Any]]]:
        """Process images sequentially and save individual page images"""
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        for page_num, image in enumerate(images):
            print(f"Processing page {page_num + 1}...")
            
            # Save image
            image_path = self._save_image(image, file_path, page_num + 1, "page")
            
            # Extract text content
            text_content = self.extract_content_from_image(image, "text")
            if text_content.strip() and "no text content found" not in text_content.lower():
                processed_content["text"].append({
                    "content": text_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "text",
                    "image_path": image_path
                })
            
            # Extract table content
            table_content = self.extract_content_from_image(image, "table")
            if table_content.strip() and "no tables found" not in table_content.lower():
                processed_content["tables"].append({
                    "content": table_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "table",
                    "image_path": image_path
                })
            
            # Extract visual content
            visual_content = self.extract_content_from_image(image, "visual")
            if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                processed_content["visuals"].append({
                    "content": visual_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "visual",
                    "image_path": image_path
                })
        
        return processed_content
    
    def process_document(self, file_path: str, force_reprocess: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Process document and extract all content types with caching"""
        
        # Check for cached extraction first
        if not force_reprocess:
            saved_content = self._load_saved_extraction(file_path)
            if saved_content is not None:
                return saved_content
        
        print(f"Processing document: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle different file types
        if file_ext == '.eml':
            processed_content = self.process_eml_document(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            processed_content = self.process_excel_document(file_path)
        elif file_ext == '.pdf':
            images = self.convert_pdf_to_images(file_path)
            processed_content = self._process_images_sequential(file_path, images)
        elif file_ext == '.pptx':
            images = self.convert_pptx_to_images(file_path)
            processed_content = self._process_images_sequential(file_path, images)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Save extraction results
        self._save_extraction_results(file_path, processed_content)
        
        return processed_content
    
    def load_from_saved_extraction(self, file_path: str):
        """Public method to load from saved extraction"""
        return self._load_saved_extraction(file_path)
    
    def get_all_saved_extractions(self):
        """Get list of all saved extractions with metadata"""
        extractions = []
        
        if not os.path.exists(self.content_dir):
            return extractions
        
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    extractions.append({
                        'extraction_file': filepath,
                        'source_file': data.get('source_file', ''),
                        'file_hash': data.get('file_hash', ''),
                        'extraction_date': data.get('extraction_date', ''),
                        'content_stats': {
                            'text_items': len(data.get('content', {}).get('text', [])),
                            'table_items': len(data.get('content', {}).get('tables', [])),
                            'visual_items': len(data.get('content', {}).get('visuals', []))
                        }
                    })
                except Exception as e:
                    print(f"Error reading extraction file {filename}: {e}")
        
        return extractions
