import fitz  # PyMuPDF
from pdf2image import convert_from_path
import os
from typing import List, Tuple
import logging

class PDFProcessor:
    """Handle PDF to image conversion and basic text extraction"""
    
    def __init__(self, output_dir: str = "images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """Convert PDF pages to images"""
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            images = convert_from_path(pdf_path, dpi=dpi)
            image_paths = []
            
            for i, image in enumerate(images):
                image_path = os.path.join(self.output_dir, f"{pdf_name}_page_{i+1}.png")
                image.save(image_path, "PNG")
                image_paths.append(image_path)
                self.logger.info(f"Saved page {i+1} to {image_path}")
            
            return image_paths
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get basic PDF information"""
        try:
            doc = fitz.open(pdf_path)
            info = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'pages': len(doc),
                'filename': os.path.basename(pdf_path)
            }
            doc.close()
            return info
        except Exception as e:
            self.logger.error(f"Error getting PDF info: {str(e)}")
            return {}