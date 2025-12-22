#!/usr/bin/env python3
"""
Enhanced OCR Extraction Script with Structure Detection
========================================================

NEW FEATURES vs original:
1. ✅ Detects and preserves document structure (headings, sections, lists)
2. ✅ Smart chunking that respects sections and doesn't break mid-paragraph
3. ✅ Extracts metadata (section numbers, document codes, has tables/figures)
4. ✅ Better handling of tables and structured content
5. ✅ OCR quality validation with confidence scores
6. ✅ TABLE EXTRACTION using pdfplumber (preserves table structure)
7. ✅ Figure caption linking to nearby content
8. ✅ Multi-column layout handling with reading order correction

Usage:
  python enhanced_ocr_extraction.py pdfs/ --output enhanced_output.json
"""

from paddleocr import PaddleOCR
import numpy as np
import sys
import os
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm
import gc
import cv2
from PIL import Image
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not installed. Table extraction disabled. Install with: pip install pdfplumber")

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ocr_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
MIN_WORDS_FOR_PAGE = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SHORT_SIDE = 1200

# Pre-built blacklist
TEXT_BLACKLIST = frozenset({
    "", " ", "i", "ii", "iii", "iv", "v", 
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "a", "b", "c", "d", "e", ".", ",", "-", "_", "|"
})

# Regex patterns compiled once
SINGLE_LETTER_PATTERN = re.compile(r'^[a-zA-Z]{1,2}$')
SINGLE_DIGIT_PATTERN = re.compile(r'^\d$')
# Multiple header patterns for better detection
SECTION_HEADER_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,100})$', re.MULTILINE),  # "1.1 Title" (extended length)
    re.compile(r'^(Appendix\s+[A-Z0-9])\s*[:\-]?\s*(.{3,80})$', re.MULTILINE | re.IGNORECASE),  # "Appendix A: Title"
    re.compile(r'^([A-Z][A-Z\s]{5,60})$', re.MULTILINE),  # "ALL CAPS HEADER"
]
SECTION_HEADER_PATTERN = SECTION_HEADER_PATTERNS[0]  # Keep for backward compat
DOC_CODE_PATTERN = re.compile(r'[A-Z]{2,4}-[A-Z]{2,4}-[A-Z]{2,4}-\d{3}')
TABLE_MARKER = re.compile(r'Table\s+\d+\.?\d*', re.IGNORECASE)
FIGURE_MARKER = re.compile(r'Figure\s+\d+\.?\d*', re.IGNORECASE)
APPENDIX_PATTERN = re.compile(r'^Appendix\s+[A-Z0-9]', re.MULTILINE)
# Document reference pattern (catches references like RS-COR-IS-005, WI-ENG-LCP-701)
DOC_REF_PATTERN = re.compile(r'[A-Z]{2,4}-[A-Z]{2,4}-[A-Z]{2,4}-\d{2,4}')


def is_valid_text(text: str) -> bool:
    """Fast text validation"""
    if not text:
        return False
    t = text.strip().lower()
    if len(t) < 3:
        return False
    if t in TEXT_BLACKLIST:
        return False
    if SINGLE_LETTER_PATTERN.match(text):
        return False
    if SINGLE_DIGIT_PATTERN.match(text):
        return False
    return True


def resize_if_large(img_bgr: np.ndarray, max_short_side: int = MAX_SHORT_SIDE) -> np.ndarray:
    """Resize image if too large"""
    h, w = img_bgr.shape[:2]
    short_side = min(h, w)
    if short_side > max_short_side:
        ratio = max_short_side / short_side
        new_w, new_h = int(w * ratio), int(h * ratio)
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr


# =========================================================
# NEW: STRUCTURE DETECTION
# =========================================================

def detect_section_headers(text: str) -> List[Tuple[str, str, int]]:
    """
    Detect section headers like "3.1 Process to Assure Cyber Security"
    Returns: List of (section_number, header_text, position)
    """
    headers = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        section_num = match.group(1)
        header_text = match.group(2)
        position = match.start()
        headers.append((section_num, header_text, position))
    return headers


def extract_document_code(text: str) -> Optional[str]:
    """Extract document codes like BN-NET-GOV-501"""
    match = DOC_CODE_PATTERN.search(text)
    return match.group(0) if match else None


def detect_structural_elements(text: str) -> Dict:
    """
    Detect structural elements in document
    Returns dict with: has_table, has_figure, has_appendix, doc_code, doc_references
    """
    # Find all document references (e.g., RS-COR-IS-005, WI-ENG-LCP-701)
    doc_refs = list(set(DOC_REF_PATTERN.findall(text)))
    
    return {
        'has_table': bool(TABLE_MARKER.search(text)),
        'has_figure': bool(FIGURE_MARKER.search(text)),
        'has_appendix': bool(APPENDIX_PATTERN.search(text)),
        'doc_code': extract_document_code(text),
        'doc_references': doc_refs,  # List of referenced documents
        'table_count': len(TABLE_MARKER.findall(text)),
        'figure_count': len(FIGURE_MARKER.findall(text)),
    }


def is_likely_header(text: str, next_text: str = None) -> bool:
    """
    Heuristic to detect if text is a header:
    - Starts with number (1, 1.1, 3.2.1)
    - All caps or Title Case
    - Short (< 100 chars)
    - Followed by longer paragraph
    """
    text = text.strip()
    
    # Check for numbered sections
    if re.match(r'^\d+(?:\.\d+)*\s+[A-Z]', text):
        return True
    
    # Check for appendix headers
    if re.match(r'^Appendix\s+[A-Z0-9]', text):
        return True
    
    # All caps short text (likely header)
    if text.isupper() and len(text) < 80 and len(text.split()) < 10:
        return True
    
    # Title case with short length
    if text.istitle() and len(text) < 80:
        return True
    
    return False


# =========================================================
# NEW: TABLE EXTRACTION WITH PDFPLUMBER
# =========================================================

def extract_tables_from_pdf(pdf_path: str) -> Dict[int, List[Dict]]:
    """
    Extract tables from PDF using pdfplumber.
    Returns: Dict mapping page_num -> list of table dicts
    
    Each table dict contains:
    - 'text': Formatted table as text (markdown-style)
    - 'bbox': Bounding box (x0, y0, x1, y1)
    - 'rows': Number of rows
    - 'cols': Number of columns
    - 'caption': Detected table caption if nearby
    """
    if not PDFPLUMBER_AVAILABLE:
        return {}
    
    tables_by_page = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = []
                
                # Extract tables from this page
                tables = page.extract_tables()
                
                if not tables:
                    continue
                
                # Get page text to find captions
                page_text = page.extract_text() or ""
                
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:  # Skip empty/single-row tables
                        continue
                    
                    # Format table as markdown-style text
                    formatted_rows = []
                    max_cols = max(len(row) for row in table if row)
                    
                    for row_idx, row in enumerate(table):
                        if not row:
                            continue
                        
                        # Clean cells
                        cells = []
                        for cell in row:
                            cell_text = str(cell).strip() if cell else ""
                            cell_text = cell_text.replace('\n', ' ')
                            cells.append(cell_text)
                        
                        # Pad row if needed
                        while len(cells) < max_cols:
                            cells.append("")
                        
                        formatted_rows.append(" | ".join(cells))
                        
                        # Add separator after header row
                        if row_idx == 0:
                            formatted_rows.append("-" * 50)
                    
                    table_text = "\n".join(formatted_rows)
                    
                    # Try to find table caption
                    caption = find_table_caption(page_text, table_idx + 1)
                    
                    # Get table bounding box if available
                    try:
                        table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                        table_finder = page.find_tables(table_settings)
                        bbox = table_finder[table_idx].bbox if table_idx < len(table_finder) else None
                    except:
                        bbox = None
                    
                    page_tables.append({
                        'text': table_text,
                        'caption': caption,
                        'rows': len(table),
                        'cols': max_cols,
                        'bbox': bbox,
                        'table_index': table_idx
                    })
                
                if page_tables:
                    tables_by_page[page_num] = page_tables
                    
    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}")
    
    return tables_by_page


def find_table_caption(page_text: str, table_num: int) -> Optional[str]:
    """
    Find caption for a table (e.g., 'Table 1: Description...')
    """
    # Look for patterns like "Table 1:", "Table 1.", "Table 1 -"
    patterns = [
        rf'(Table\s+{table_num}[.:]\s*[^\n]{{5,100}})',
        rf'(Table\s+{table_num}\s*[-–]\s*[^\n]{{5,100}})',
        rf'(Table\s+{table_num}\s+[A-Z][^\n]{{5,80}})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def format_table_as_chunk(table_data: Dict, page_num: int, file_name: str) -> Dict:
    """
    Format extracted table as a chunk with proper metadata.
    """
    caption = table_data.get('caption', '')
    table_text = table_data.get('text', '')
    
    # Create searchable chunk text with caption
    if caption:
        chunk_text = f"{caption}\n\n{table_text}"
    else:
        chunk_text = f"[Table with {table_data['rows']} rows x {table_data['cols']} columns]\n\n{table_text}"
    
    return {
        'text': chunk_text,
        'content_type': 'table',
        'page_num': page_num,
        'caption': caption,
        'rows': table_data['rows'],
        'cols': table_data['cols'],
    }


# =========================================================
# NEW: FIGURE CAPTION EXTRACTION AND LINKING
# =========================================================

FIGURE_CAPTION_PATTERN = re.compile(
    r'(Figure\s+\d+(?:\.\d+)?[.:]\s*[^\n]{5,150})',
    re.IGNORECASE
)

def extract_figure_captions(text: str) -> List[Dict]:
    """
    Extract figure captions from text.
    Returns list of {'caption': str, 'figure_num': str, 'position': int}
    """
    captions = []
    
    for match in FIGURE_CAPTION_PATTERN.finditer(text):
        caption_text = match.group(1).strip()
        
        # Extract figure number
        fig_num_match = re.search(r'Figure\s+(\d+(?:\.\d+)?)', caption_text, re.IGNORECASE)
        fig_num = fig_num_match.group(1) if fig_num_match else "unknown"
        
        captions.append({
            'caption': caption_text,
            'figure_num': fig_num,
            'position': match.start()
        })
    
    return captions


def link_figures_to_context(text: str, chunk_size: int = 500) -> List[Dict]:
    """
    Extract figure captions with surrounding context for better search.
    Returns list of figure chunks with context.
    """
    figure_chunks = []
    captions = extract_figure_captions(text)
    
    for cap_info in captions:
        pos = cap_info['position']
        caption = cap_info['caption']
        
        # Get context before and after the caption
        context_start = max(0, pos - chunk_size // 2)
        context_end = min(len(text), pos + len(caption) + chunk_size // 2)
        
        # Try to align to sentence boundaries
        before_text = text[context_start:pos]
        after_text = text[pos + len(caption):context_end]
        
        # Find sentence start
        sentence_start = before_text.rfind('. ')
        if sentence_start != -1:
            before_text = before_text[sentence_start + 2:]
        
        # Find sentence end
        sentence_end = after_text.find('. ')
        if sentence_end != -1:
            after_text = after_text[:sentence_end + 1]
        
        context_text = f"{before_text.strip()} {caption} {after_text.strip()}"
        
        figure_chunks.append({
            'text': context_text.strip(),
            'content_type': 'figure',
            'caption': caption,
            'figure_num': cap_info['figure_num'],
        })
    
    return figure_chunks


# =========================================================
# NEW: READING ORDER CORRECTION FOR MULTI-COLUMN LAYOUTS
# =========================================================

def sort_by_reading_order(detections: List[Dict], page_width: int = None) -> List[Dict]:
    """
    Sort OCR detections by reading order.
    Handles multi-column layouts by detecting columns and sorting within each.
    
    Args:
        detections: List of dicts with 'text', 'bbox' (x0, y0, x1, y1)
        page_width: Page width to detect column boundaries
    
    Returns:
        Sorted list of detections in reading order
    """
    if not detections:
        return detections
    
    # If no bbox info, return as-is (already in OCR reading order)
    if 'bbox' not in detections[0]:
        return detections
    
    # Detect if multi-column based on x-positions
    x_positions = [d['bbox'][0] for d in detections if 'bbox' in d]
    
    if not x_positions:
        return detections
    
    # Use clustering to detect columns
    x_positions_sorted = sorted(set(x_positions))
    
    # Simple heuristic: if there's a gap > 30% of page width, it's multi-column
    if page_width is None:
        page_width = max(d['bbox'][2] for d in detections if 'bbox' in d)
    
    column_gap_threshold = page_width * 0.15  # 15% gap indicates column
    
    columns = [[]]
    prev_x = x_positions_sorted[0] if x_positions_sorted else 0
    
    for x in x_positions_sorted:
        if x - prev_x > column_gap_threshold:
            columns.append([])
        columns[-1].append(x)
        prev_x = x
    
    # If 2+ columns detected, sort by column then by Y position
    if len(columns) >= 2:
        column_boundaries = []
        for col in columns:
            if col:
                column_boundaries.append((min(col), max(col)))
        
        def get_column_index(bbox):
            x = bbox[0]
            for i, (col_min, col_max) in enumerate(column_boundaries):
                if col_min - 50 <= x <= col_max + 50:
                    return i
            return 0
        
        # Sort by: column index first, then Y position
        detections_sorted = sorted(
            detections,
            key=lambda d: (get_column_index(d.get('bbox', [0,0,0,0])), 
                          d.get('bbox', [0,0,0,0])[1])  # y0
        )
        return detections_sorted
    
    # Single column - just sort by Y position
    return sorted(detections, key=lambda d: d.get('bbox', [0,0,0,0])[1])


def split_into_sections(text: str) -> List[Dict]:
    """
    Split text into logical sections based on headers
    Returns: List of dicts with 'header', 'content', 'section_num'
    """
    headers = detect_section_headers(text)
    
    if not headers:
        # No clear sections found, return as single section
        return [{
            'header': None,
            'section_num': None,
            'content': text,
            'start_pos': 0,
            'end_pos': len(text)
        }]
    
    sections = []
    for i, (section_num, header_text, pos) in enumerate(headers):
        # Find where this section ends (start of next section or end of text)
        if i < len(headers) - 1:
            end_pos = headers[i + 1][2]
        else:
            end_pos = len(text)
        
        # Extract content for this section
        content = text[pos:end_pos].strip()
        
        sections.append({
            'header': header_text,
            'section_num': section_num,
            'content': content,
            'start_pos': pos,
            'end_pos': end_pos
        })
    
    return sections


# =========================================================
# NEW: SMART CHUNKING WITH STRUCTURE AWARENESS
# =========================================================

def chunk_page_text(page_text: str, page_num: int, chunk_size: int = CHUNK_SIZE, 
                    overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunk a single page's text while respecting structure.
    Each chunk gets the correct page number directly.
    
    Returns: List of dicts with 'text', 'section_num', 'header', 'page_num', 'continues_to_next'
    """
    if not page_text or not page_text.strip():
        return []
    
    # Detect sections within this page
    sections = split_into_sections(page_text)
    
    chunks = []
    
    for section in sections:
        section_num = section.get('section_num')
        section_header = section.get('header')
        section_content = section.get('content', '')
        
        # If section fits in one chunk, keep it together
        if len(section_content) <= chunk_size:
            if section_header:
                chunk_text = f"{section_num} {section_header}\n\n{section_content}"
            else:
                chunk_text = section_content
            
            chunks.append({
                'text': chunk_text.strip(),
                'section_num': section_num,
                'header': section_header,
                'page_num': page_num,
                'is_complete_section': True,
                'continues_to_next': False
            })
            continue
        
        # Section too large - split by paragraphs
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            continue
        
        current_chunk_text = []
        current_length = 0
        
        # Always start with section header
        if section_header:
            header_line = f"{section_num} {section_header}\n\n"
            current_chunk_text.append(header_line)
            current_length += len(header_line)
        
        for para_idx, para in enumerate(paragraphs):
            para_len = len(para)
            
            # If adding this para exceeds chunk size
            if current_length + para_len > chunk_size and current_chunk_text:
                # Save current chunk
                chunks.append({
                    'text': ''.join(current_chunk_text).strip(),
                    'section_num': section_num,
                    'header': section_header,
                    'page_num': page_num,
                    'is_complete_section': False,
                    'continues_to_next': False
                })
                
                # Get overlap content
                overlap_content = []
                overlap_size = 0
                for prev_para in reversed(current_chunk_text):
                    if overlap_size + len(prev_para) <= overlap:
                        overlap_content.insert(0, prev_para)
                        overlap_size += len(prev_para)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_text = []
                current_length = 0
                
                if section_header:
                    header_line = f"{section_num} {section_header}\n\n"
                    current_chunk_text.append(header_line)
                    current_length += len(header_line)
                
                for overlap_para in overlap_content:
                    current_chunk_text.append(overlap_para)
                    current_length += len(overlap_para)
            
            current_chunk_text.append(para + '\n\n')
            current_length += para_len
        
        # Save remaining chunk
        if current_chunk_text:
            chunks.append({
                'text': ''.join(current_chunk_text).strip(),
                'section_num': section_num,
                'header': section_header,
                'page_num': page_num,
                'is_complete_section': False,
                'continues_to_next': False
            })
    
    return chunks


def chunk_with_cross_page_awareness(pages: List[Dict], chunk_size: int = CHUNK_SIZE, 
                                    overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunk text page-by-page, tracking when content spans pages.
    
    Args:
        pages: List of {'page_num': int, 'text': str, ...} from OCR
        chunk_size: Max chars per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of chunks with accurate page_num and cross-page info
    """
    all_chunks = []
    
    for page_idx, page in enumerate(pages):
        page_num = page['page_num']
        page_text = page['text']
        
        # Get chunks from this page
        page_chunks = chunk_page_text(page_text, page_num, chunk_size, overlap)
        
        # Check if content might continue from previous page
        # (starts mid-sentence or lowercase)
        if page_idx > 0 and page_text.strip():
            first_char = page_text.strip()[0]
            starts_mid_sentence = first_char.islower() or first_char in ',;'
            
            if starts_mid_sentence and page_chunks:
                page_chunks[0]['continues_from_previous'] = True
        
        # Check if last chunk might continue to next page
        # (ends mid-sentence)
        if page_chunks and page_text.strip():
            last_text = page_text.strip()
            ends_mid_sentence = not last_text[-1] in '.!?"\''
            
            if ends_mid_sentence:
                page_chunks[-1]['continues_to_next'] = True
        
        all_chunks.extend(page_chunks)
    
    return all_chunks


# =========================================================
# ENHANCED OCR CLASS
# =========================================================

class EnhancedTextDetection:
    def __init__(self) -> None:
        self.ocr = None
        self.stats = {
            'files': 0, 
            'pages': 0, 
            'words': 0,
            'low_confidence_pages': 0,
            'avg_confidence': []
        }

    def initPaddle(self, det_limit=1920, use_gpu=True) -> None:
        """Initialize PaddleOCR"""
        self.ocr = PaddleOCR(
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_type='max',
            text_det_limit_side_len=det_limit,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_recognition_batch_size=16,
        )
        logger.info(f"✓ PaddleOCR initialized on {'GPU' if use_gpu else 'CPU'}")

    def split_into_tiles(self, img, num_tiles=2, overlap=50):
        """Split image into tiles"""
        h, w = img.shape[:2]
        
        if num_tiles == 1:
            return [(img, 0, 0)]
        elif num_tiles == 2:
            mid_y = h // 2
            return [
                (img[0:mid_y + overlap, :], 0, 0),
                (img[mid_y - overlap:h, :], 0, mid_y - overlap)
            ]
        elif num_tiles == 4:
            mid_y, mid_x = h // 2, w // 2
            return [
                (img[0:mid_y + overlap, 0:mid_x + overlap], 0, 0),
                (img[0:mid_y + overlap, mid_x - overlap:w], mid_x - overlap, 0),
                (img[mid_y - overlap:h, 0:mid_x + overlap], 0, mid_y - overlap),
                (img[mid_y - overlap:h, mid_x - overlap:w], mid_x - overlap, mid_y - overlap)
            ]
        else:
            raise ValueError(f"num_tiles must be 1, 2 or 4")

    def process_tile(self, tile_data):
        """Process single tile"""
        tile, offset_x, offset_y, tile_idx = tile_data
        result = self.ocr.predict(tile)
        return result, offset_x, offset_y, tile_idx

    def adjust_coordinates(self, poly, offset_x, offset_y):
        """Adjust coordinates to full image space"""
        return [[float(pt[0]) + offset_x, float(pt[1]) + offset_y] for pt in poly]

    def merge_results(self, tile_results, overlap, num_tiles):
        """Merge results from tiles with bounding box info for reading order"""
        merged = []
        confidences = []
        
        for result, offset_x, offset_y, tile_idx in tile_results:
            if not result or not result[0]:
                continue
            
            for res in result:
                texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                scores = res.get('rec_scores', []) if isinstance(res, dict) else getattr(res, 'rec_scores', [])
                polys = res.get('rec_polys', []) if isinstance(res, dict) else getattr(res, 'rec_polys', [])
                
                if not texts:
                    continue
                
                for j, text in enumerate(texts):
                    if not text or not str(text).strip():
                        continue
                    
                    score = scores[j] if j < len(scores) else 0.0
                    poly = polys[j] if j < len(polys) else None
                    
                    if poly is None:
                        continue
                    
                    adjusted_box = self.adjust_coordinates(poly, offset_x, offset_y)
                    
                    # Skip overlap regions
                    if num_tiles > 1 and tile_idx > 0:
                        center_y = sum(pt[1] for pt in adjusted_box) / 4
                        if num_tiles == 2 and tile_idx == 1 and center_y < offset_y + overlap:
                            continue
                        if num_tiles == 4:
                            center_x = sum(pt[0] for pt in adjusted_box) / 4
                            if tile_idx == 1 and center_x < offset_x + overlap:
                                continue
                            if tile_idx == 2 and center_y < offset_y + overlap:
                                continue
                            if tile_idx == 3 and (center_x < offset_x + overlap or center_y < offset_y + overlap):
                                continue
                    
                    # Calculate bounding box from polygon (x0, y0, x1, y1)
                    xs = [pt[0] for pt in adjusted_box]
                    ys = [pt[1] for pt in adjusted_box]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    
                    merged.append({
                        "text": str(text),
                        "confidence": float(score),
                        "bbox": bbox,  # Include bbox for reading order sorting
                    })
                    confidences.append(float(score))
        
        return merged, confidences

    def process_pdf_pages(self, file_path: str, num_tiles: int = 2, dpi: int = 200, 
                          first_page_only: bool = False, resize_large: bool = True) -> List[Dict]:
        """Process PDF pages and return structured text"""
        try:
            images = convert_from_path(file_path, dpi=dpi)
        except Exception as e:
            logger.error(f"Failed to convert PDF {file_path}: {e}")
            return []
        
        pages_data = []
        num_images = len(images)
        pages_to_process = [1] if first_page_only else range(1, num_images + 1)
        
        for page_num in pages_to_process:
            pil_image = images[page_num - 1]
            img = np.array(pil_image)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            page_width = img_bgr.shape[1]  # Get page width for column detection
            
            if resize_large:
                img_bgr = resize_if_large(img_bgr, MAX_SHORT_SIDE)
            
            tiles = self.split_into_tiles(img_bgr, num_tiles, overlap=50)
            tile_data = [(tile, ox, oy, i) for i, (tile, ox, oy) in enumerate(tiles)]
            
            tile_results = [self.process_tile(td) for td in tile_data]
            merged_results, confidences = self.merge_results(tile_results, 50, num_tiles)
            
            # Calculate average confidence for this page
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            self.stats['avg_confidence'].append(avg_confidence)
            
            # Flag low confidence pages for manual review
            if avg_confidence < 0.7:
                self.stats['low_confidence_pages'] += 1
                logger.warning(f"Low OCR confidence on page {page_num}: {avg_confidence:.2f}")
            
            # NEW: Sort by reading order (handles multi-column layouts)
            sorted_results = sort_by_reading_order(merged_results, page_width)
            
            valid_detections = [
                detection for detection in sorted_results 
                if is_valid_text(detection.get("text", "").strip())
            ]
            
            # Join texts - use double newline between groups for paragraph structure
            # Group consecutive short lines (likely lists/tables) vs paragraphs
            grouped_text = []
            current_group = []
            
            for detection in valid_detections:
                text = detection["text"].strip()
                current_group.append(text)
                # If text ends with sentence-ending punctuation, close the group
                if text.rstrip().endswith(('.', '!', '?', ':')):
                    grouped_text.append(' '.join(current_group))
                    current_group = []
            
            # Don't forget remaining text
            if current_group:
                grouped_text.append(' '.join(current_group))
            
            page_text = '\n\n'.join(grouped_text)
            word_count = len(page_text.split())
            
            if word_count >= MIN_WORDS_FOR_PAGE:
                pages_data.append({
                    'page_num': page_num,
                    'text': page_text,
                    'word_count': word_count,
                    'avg_confidence': avg_confidence,
                    'low_confidence_warning': avg_confidence < 0.7
                })
            
            self.stats['pages'] += 1
            self.stats['words'] += word_count
            
            del img, img_bgr, tiles, tile_data, tile_results, merged_results, pil_image
        
        del images
        gc.collect()
        
        return pages_data


# =========================================================
# ENHANCED BUILD CHUNKS FUNCTION
# =========================================================

def build_enhanced_chunks(pdf_path: str, source_folder: str, ocr_engine: EnhancedTextDetection,
                         chunk_size: int, overlap: int, num_tiles: int, dpi: int,
                         first_page_only: bool = False, resize_large: bool = True,
                         extract_tables: bool = True) -> List[Dict]:
    """
    Build enhanced chunks with structure detection, table extraction, and figure linking.
    
    NEW FEATURES:
    - Extracts tables separately using pdfplumber (preserves structure)
    - Links figure captions to surrounding context
    - Adds content_type field: 'text', 'table', 'figure'
    """
    
    file_base = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    
    # ===== STEP 1: Extract tables with pdfplumber =====
    table_chunks = []
    tables_by_page = {}
    
    if extract_tables and PDFPLUMBER_AVAILABLE:
        try:
            tables_by_page = extract_tables_from_pdf(pdf_path)
            
            for page_num, page_tables in tables_by_page.items():
                for table_idx, table_data in enumerate(page_tables):
                    table_chunk = format_table_as_chunk(table_data, page_num, file_base)
                    
                    chunk_id = f"{file_base}__table_p{page_num}_t{table_idx}"
                    
                    table_chunks.append({
                        "chunk_id": chunk_id,
                        "file_name": file_base,
                        "page_number": page_num,
                        "chunk_index": f"table_{table_idx}",
                        "chunk_text": table_chunk['text'],
                        "word_count": len(table_chunk['text'].split()),
                        "content_type": "table",
                        "table_caption": table_chunk.get('caption'),
                        "table_rows": table_chunk.get('rows'),
                        "table_cols": table_chunk.get('cols'),
                        "section_number": None,
                        "section_header": None,
                        "is_complete_section": True,
                        "source_folder": source_folder,
                        "doc_title": file_base,
                        "text_source": "pdfplumber_table",
                        "created_date": datetime.now().isoformat()
                    })
            
            if table_chunks:
                logger.info(f"  ✓ Extracted {len(table_chunks)} tables from {pdf_path}")
                
        except Exception as e:
            logger.warning(f"Table extraction failed for {pdf_path}: {e}")
    
    # ===== STEP 2: OCR for regular text =====
    pages = ocr_engine.process_pdf_pages(pdf_path, num_tiles=num_tiles, dpi=dpi, 
                                        first_page_only=first_page_only,
                                        resize_large=resize_large)
    
    if not pages:
        return table_chunks  # Return any tables we found
    
    # Combine all pages into full text
    full_text = '\n\n'.join(page['text'] for page in pages)
    
    # Detect document structure
    structure = detect_structural_elements(full_text)
    doc_code = structure.get('doc_code')
    
    # Build page offset map for figure page detection
    page_offsets = []  # List of (start_offset, end_offset, page_num)
    current_offset = 0
    for page in pages:
        page_len = len(page['text']) + 2  # +2 for '\n\n' separator
        page_offsets.append((current_offset, current_offset + page_len, page['page_num']))
        current_offset += page_len
    
    def find_page_for_position(pos: int) -> int:
        """Find which page a text position belongs to"""
        for start, end, pnum in page_offsets:
            if start <= pos < end:
                return pnum
        return 1  # Default to page 1
    
    # ===== STEP 3: Extract figure captions with context =====
    figure_chunks = []
    figure_captions = extract_figure_captions(full_text)
    
    for fig_idx, fig_info in enumerate(figure_captions):
        # Get context around figure
        fig_context = link_figures_to_context(full_text)
        
        for ctx_idx, ctx in enumerate(fig_context):
            if ctx.get('figure_num') == fig_info['figure_num']:
                chunk_id = f"{file_base}__figure_{fig_info['figure_num']}"
                
                # Find page number from the figure caption's position
                fig_page = find_page_for_position(fig_info.get('position', 0))
                
                figure_chunks.append({
                    "chunk_id": chunk_id,
                    "file_name": file_base,
                    "page_number": fig_page,
                    "chunk_index": f"figure_{fig_idx}",
                    "chunk_text": ctx['text'],
                    "word_count": len(ctx['text'].split()),
                    "content_type": "figure",
                    "figure_caption": ctx.get('caption'),
                    "figure_number": ctx.get('figure_num'),
                    "section_number": None,
                    "section_header": None,
                    "is_complete_section": True,
                    "document_code": doc_code,
                    "source_folder": source_folder,
                    "doc_title": file_base,
                    "text_source": "ocr_figure_context",
                    "created_date": datetime.now().isoformat()
                })
                break  # Only add once per figure
    
    if figure_chunks:
        logger.info(f"  ✓ Extracted {len(figure_chunks)} figure contexts from {pdf_path}")
    
    # ===== STEP 4: Page-by-page smart chunking =====
    # This is the key change: chunk each page individually, so page numbers are always accurate
    structured_chunks = chunk_with_cross_page_awareness(pages, chunk_size, overlap)
    
    # Build output chunks with enhanced metadata
    text_chunks = []
    
    for idx, chunk_info in enumerate(structured_chunks):
        chunk_id = f"{file_base}__c{idx}"
        chunk_text = chunk_info['text']
        page_num = chunk_info['page_num']  # Directly from page-aware chunking
        
        # Determine content type based on content
        content_type = "text"
        if TABLE_MARKER.search(chunk_text):
            content_type = "text_with_table_ref"
        elif FIGURE_MARKER.search(chunk_text):
            content_type = "text_with_figure_ref"
        
        chunk_data = {
            "chunk_id": chunk_id,
            "file_name": file_base,
            "page_number": page_num,
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "word_count": len(chunk_text.split()),
            "content_type": content_type,
            
            # Structure metadata
            "section_number": chunk_info.get('section_num'),
            "section_header": chunk_info.get('header'),
            "is_complete_section": chunk_info.get('is_complete_section', False),
            
            # Cross-page info (helpful for RAG context)
            "continues_from_previous": chunk_info.get('continues_from_previous', False),
            "continues_to_next": chunk_info.get('continues_to_next', False),
            
            # Document metadata
            "document_code": doc_code,
            "has_table": structure['has_table'],
            "has_figure": structure['has_figure'],
            "has_appendix": structure['has_appendix'],
            "doc_references": structure.get('doc_references', []),
            
            # Original metadata
            "source_folder": source_folder,
            "doc_title": file_base,
            "text_source": "paddleocr_v5_enhanced",
            "created_date": datetime.now().isoformat()
        }
        
        text_chunks.append(chunk_data)
    
    # ===== STEP 5: Combine all chunks =====
    # Order: tables first (often referenced), then text, then figures
    all_chunks = table_chunks + text_chunks + figure_chunks
    
    # Re-index all chunks sequentially
    for i, chunk in enumerate(all_chunks):
        chunk['chunk_index'] = i
    
    return all_chunks


# =========================================================
# MAIN FUNCTION
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enhanced OCR with Structure Detection")
    
    parser.add_argument("folder_path", help="Path to PDF folder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: <folder>_enhanced_output.json)")
    parser.add_argument("--first-page-only", action="store_true",
                        help="Process only first page of each PDF")
    parser.add_argument("--no-resize", action="store_true",
                        help="Disable image resizing")
    parser.add_argument("--no-tables", action="store_true",
                        help="Disable table extraction with pdfplumber")
    parser.add_argument("--tiles", type=int, choices=[1, 2, 4], default=1,
                        help="Number of tiles for OCR (default: 2)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")
    parser.add_argument("--det-limit", type=int, default=1920, help="OCR detection limit")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, 
                        help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, 
                        help="Chunk overlap in characters")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    return parser.parse_args()
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    folder = args.folder_path
    
    if not os.path.isdir(folder):
        logger.error(f"Invalid folder path: {folder}")
        sys.exit(1)
    
    # Output filename
    if args.output:
        output_file = args.output
    else:
        folder_name = os.path.basename(os.path.normpath(folder))
        output_file = f"{folder_name}_enhanced_output.json"
    
    # Find PDFs
    pdf_files = []
    for root, dirs, files in os.walk(folder):
        source = os.path.basename(root) if root != folder else os.path.basename(folder)
        for f in files:
            if f.lower().endswith(".pdf") and not f.endswith('.Zone.Identifier'):
                pdf_files.append((os.path.join(root, f), source))
    
    if not pdf_files:
        logger.error("No PDFs found.")
        sys.exit(1)
    
    pdf_files.sort()
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info(f"✨ Enhanced mode: Structure detection + smart chunking ENABLED")
    
    # Check table extraction availability
    extract_tables = not args.no_tables and PDFPLUMBER_AVAILABLE
    if extract_tables:
        logger.info(f"✓ Table extraction ENABLED (pdfplumber)")
    else:
        if args.no_tables:
            logger.info(f"⚠️ Table extraction DISABLED (--no-tables flag)")
        else:
            logger.info(f"⚠️ Table extraction DISABLED (pdfplumber not installed)")
    
    # Initialize enhanced OCR engine
    logger.info("Initializing Enhanced PaddleOCR...")
    ocr_engine = EnhancedTextDetection()
    ocr_engine.initPaddle(det_limit=args.det_limit, use_gpu=not args.cpu)
    
    # Process PDFs
    all_chunks = []
    start_time = datetime.now()
    
    for idx, (pdf_path, source) in enumerate(tqdm(pdf_files, desc="Processing PDFs"), 1):
        chunks = build_enhanced_chunks(
            pdf_path, source, ocr_engine,
            args.chunk_size, args.overlap, args.tiles, args.dpi,
            first_page_only=args.first_page_only,
            resize_large=not args.no_resize,
            extract_tables=extract_tables
        )
        if chunks:
            all_chunks.extend(chunks)
        
        if idx % 10 == 0:
            gc.collect()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Calculate OCR quality stats
    avg_conf = sum(ocr_engine.stats['avg_confidence']) / len(ocr_engine.stats['avg_confidence']) \
               if ocr_engine.stats['avg_confidence'] else 0.0
    
    # Calculate content type stats
    table_chunks = sum(1 for c in all_chunks if c.get('content_type') == 'table')
    figure_chunks = sum(1 for c in all_chunks if c.get('content_type') == 'figure')
    text_chunks = sum(1 for c in all_chunks if c.get('content_type') in ['text', 'text_with_table_ref', 'text_with_figure_ref'])
    
    # Build output
    output_data = {
        "folder_path": os.path.abspath(folder),
        "extraction_date": datetime.now().isoformat(),
        "processing_time_seconds": elapsed,
        "total_chunks": len(all_chunks),
        "total_pdfs": len(pdf_files),
        "content_types": {
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "figure_chunks": figure_chunks
        },
        "ocr_quality": {
            "average_confidence": round(avg_conf, 3),
            "low_confidence_pages": ocr_engine.stats['low_confidence_pages'],
            "total_pages": ocr_engine.stats['pages']
        },
        "config": {
            "enhanced_mode": True,
            "structure_detection": True,
            "smart_chunking": True,
            "table_extraction": extract_tables,
            "figure_linking": True,
            "reading_order_correction": True,
            "first_page_only": args.first_page_only,
            "tiles": args.tiles,
            "dpi": args.dpi,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "use_gpu": not args.cpu
        },
        "chunks": all_chunks
    }
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Stats
    sections_with_metadata = sum(1 for c in all_chunks if c.get('section_number'))
    docs_with_codes = sum(1 for c in all_chunks if c.get('document_code'))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Enhanced OCR extraction complete!")
    logger.info(f"✓ Processed {len(pdf_files)} PDFs in {elapsed:.1f}s")
    logger.info(f"✓ Extracted {len(all_chunks)} total chunks:")
    logger.info(f"    - Text chunks: {text_chunks}")
    logger.info(f"    - Table chunks: {table_chunks}")
    logger.info(f"    - Figure chunks: {figure_chunks}")
    logger.info(f"✓ OCR quality: {avg_conf:.1%} average confidence")
    logger.info(f"✓ Low confidence pages: {ocr_engine.stats['low_confidence_pages']}/{ocr_engine.stats['pages']}")
    logger.info(f"✓ Chunks with section metadata: {sections_with_metadata}/{len(all_chunks)}")
    logger.info(f"✓ Documents with codes detected: {docs_with_codes}")
    logger.info(f"✓ Output: {output_file}")
    logger.info(f"\nNext: python rag-indexer-ppocrv5-embed-separated.py {output_file}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()