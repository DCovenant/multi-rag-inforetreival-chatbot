#!/usr/bin/env python3
"""
RAG Document OCR Extraction with PPOCRv5 + Structure Preservation
=================================================================

Extracts text from PDFs using PaddleOCR's PPOCRv5 model with GPU acceleration.
Preserves document structure: titles, headers, sections, tables, body text.

Compatible with PaddleOCR 3.3.2+

Requirements:
  pip install paddlepaddle-gpu paddleocr pdf2image opencv-python numpy tqdm

Usage:
  python rag-indexer-ppocrv5.py pdfs/ --output output.json
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import logging
import gc
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

from paddleocr import PaddleOCR

# Memory optimization
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fast_eager_deletion_mode'] = 'True'
Image.MAX_IMAGE_PIXELS = 933120000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DPI = 200


@dataclass
class TextBlock:
    """Represents a detected text block."""
    text: str
    confidence: float
    line_index: int
    block_type: str = "body"  # title, section_header, subsection_header, table_header, table_cell, body
    section_number: Optional[str] = None
    level: int = 0
    bbox: Optional[List[int]] = None  # [x_min, y_min, x_max, y_max]


@dataclass 
class DocumentChunk:
    """A chunk ready for Elasticsearch indexing."""
    chunk_id: str
    file_name: str
    page_number: int
    chunk_index: int
    chunk_text: str
    word_count: int
    content_type: str
    section_hierarchy: List[str] = field(default_factory=list)
    section_number: Optional[str] = None
    parent_section: Optional[str] = None
    has_table: bool = False
    table_data: Optional[Dict] = None
    source_folder: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SectionHierarchyTracker:
    """
    Tracks section hierarchy with strict validation.
    Only updates hierarchy when encountering valid section headers.
    """
    
    # Strict patterns - must be standalone section headers, not embedded in text
    VALID_SECTION_PATTERNS = [
        # Standard numbered sections: "1 Introduction", "2 Requirements"
        # Must have: number, space, capitalized word, reasonable title length
        (r'^(\d{1,2})\s+([A-Z][a-zA-Z\s]{2,50})$', 'major'),
        
        # Subsections: "1.1 Overview", "2.3.1 Details"
        (r'^(\d{1,2}\.\d{1,2})\s+([A-Z][a-zA-Z\s]{2,60})$', 'sub'),
        (r'^(\d{1,2}\.\d{1,2}\.\d{1,2})\s+([A-Z][a-zA-Z\s]{2,60})$', 'subsub'),
        
        # Appendix patterns
        (r'^(Appendix\s+[A-Z])\s*[-:]?\s*(.{0,60})$', 'appendix'),
        (r'^(Annex\s+[A-Z0-9])\s*[-:]?\s*(.{0,60})$', 'appendix'),
        
        # Chapter/Section keywords
        (r'^(Chapter\s+\d{1,2})\s*[-:]?\s*(.{2,60})$', 'major'),
        (r'^(Section\s+\d{1,2})\s*[-:]?\s*(.{2,60})$', 'major'),
    ]
    
    # Patterns that should NOT be treated as sections (false positives)
    EXCLUDE_PATTERNS = [
        r'^\d+\s*(Hz|kHz|MHz|GHz)',  # Frequencies
        r'^\d+\s*(V|kV|mV)',          # Voltages  
        r'^\d+\s*(A|mA|kA)',          # Currents
        r'^\d+\s*(W|kW|MW)',          # Power
        r'^\d+\s*(mm|cm|m|km)',       # Length
        r'^\d+\s*(°|degrees?|C|F)',   # Temperature/angles
        r'^\d+\.\d+\.\d+\s+\w',       # IP addresses or version numbers in text
        r'^\d+\s+(of|de|from|to|and|or|the|a|an)\s',  # Numbers at start of sentences
        r'^\d+\.\d+\s+(of|de|from|to|and|or|the|a|an|is|are|was|were|shall|should|must|may)\s',  # Numbered list items in sentences
    ]
    
    def __init__(self):
        self.hierarchy = []  # List of (section_number, title) tuples
        self.last_section_number = None
    
    def is_valid_section_header(self, text: str) -> Tuple[bool, Optional[str], Optional[str], str]:
        """
        Check if text is a valid section header.
        
        Returns:
            (is_valid, section_number, title, section_type)
        """
        text = text.strip()
        
        # Quick reject: too short or too long
        if len(text) < 3 or len(text) > 80:
            return (False, None, None, '')
        
        # Quick reject: starts with lowercase
        if text[0].islower():
            return (False, None, None, '')
        
        # Check exclusion patterns first
        for pattern in self.EXCLUDE_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return (False, None, None, '')
        
        # Check valid section patterns
        for pattern, section_type in self.VALID_SECTION_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                section_num = match.group(1).strip()
                title = match.group(2).strip() if match.lastindex >= 2 else ''
                
                # Additional validation: title should have actual words
                if title and len(title.split()) < 1:
                    continue
                
                return (True, section_num, title, section_type)
        
        return (False, None, None, '')
    
    def update(self, text: str) -> Tuple[bool, Optional[str], int]:
        """
        Try to update hierarchy with text.
        
        Returns:
            (was_updated, section_number, level)
        """
        is_valid, section_num, title, section_type = self.is_valid_section_header(text)
        
        if not is_valid:
            return (False, None, 0)
        
        # Determine hierarchy level
        if section_type == 'major' or section_type == 'appendix':
            level = 1
        elif section_type == 'sub':
            level = 2
        elif section_type == 'subsub':
            level = 3
        else:
            level = section_num.count('.') + 1
        
        # Build hierarchy entry
        entry = f"{section_num} {title}".strip() if title else section_num
        
        # Update hierarchy - trim to current level and append
        self.hierarchy = self.hierarchy[:level-1]
        self.hierarchy.append(entry)
        self.last_section_number = section_num
        
        return (True, section_num, level)
    
    def get_hierarchy(self) -> List[str]:
        """Get current section hierarchy as list."""
        return self.hierarchy.copy()
    
    def get_current_section(self) -> Optional[str]:
        """Get current section number."""
        return self.last_section_number
    
    def reset(self):
        """Reset hierarchy for new document."""
        self.hierarchy = []
        self.last_section_number = None


class TableDetector:
    """
    Detects and extracts table structure using bounding box coordinates.
    Groups text blocks into rows and columns based on spatial alignment.
    """
    
    # Table header indicators
    TABLE_HEADER_PATTERNS = [
        r'^Table\s+\d+',
        r'^Tabela\s+\d+',
        r'^Tab\.\s*\d+',
        r'^TABLE\s+\d+',
    ]
    
    def __init__(self, row_tolerance: int = 15, col_tolerance: int = 30):
        """
        Args:
            row_tolerance: Y-coordinate tolerance for grouping into same row
            col_tolerance: X-coordinate tolerance for grouping into same column
        """
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance
    
    def is_table_header(self, text: str) -> bool:
        """Check if text is a table caption/header."""
        for pattern in self.TABLE_HEADER_PATTERNS:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def detect_table_region(self, blocks: List[Dict], start_idx: int) -> Tuple[int, List[Dict]]:
        """
        Detect table region starting from a table header.
        
        Returns:
            (end_index, table_blocks)
        """
        table_blocks = []
        
        if start_idx >= len(blocks):
            return start_idx, []
        
        # Get table header position
        header_block = blocks[start_idx]
        header_bbox = header_block.get('bbox')
        
        if not header_bbox:
            return start_idx + 1, []
        
        header_y = header_bbox[1]  # Top of header
        
        # Collect blocks that appear to be part of the table
        # Tables typically have consistent left margins and grid-like structure
        i = start_idx + 1
        last_y = header_y
        
        while i < len(blocks):
            block = blocks[i]
            bbox = block.get('bbox')
            
            if not bbox:
                i += 1
                continue
            
            y_top = bbox[1]
            
            # If we've moved significantly down and hit what looks like a new section, stop
            if y_top - last_y > 100:  # Large vertical gap
                # Check if next text looks like a section header or regular paragraph
                text = block.get('text', '')
                if len(text) > 100 or text[0].islower() if text else True:
                    break
            
            # Check if this looks like table content (short text, aligned)
            text = block.get('text', '')
            if len(text) > 200:  # Too long for table cell
                break
            
            table_blocks.append(block)
            last_y = y_top
            i += 1
            
            # Limit table size
            if len(table_blocks) > 50:
                break
        
        return i, table_blocks
    
    def extract_table_structure(self, blocks: List[Dict], header_text: str) -> Dict:
        """
        Extract table structure from blocks.
        
        Returns:
            {
                'table_header': str,
                'table_data': {'row,col': 'cell_text', ...},
                'rows': int,
                'cols': int
            }
        """
        if not blocks:
            return {
                'table_header': header_text,
                'table_data': {},
                'rows': 0,
                'cols': 0
            }
        
        # Group blocks by rows (similar Y coordinates)
        rows = self._group_by_rows(blocks)
        
        if not rows:
            return {
                'table_header': header_text,
                'table_data': {},
                'rows': 0,
                'cols': 0
            }
        
        # Determine column positions
        col_positions = self._find_column_positions(rows)
        
        # Build table data
        table_data = {}
        for row_idx, row_blocks in enumerate(rows):
            for block in row_blocks:
                bbox = block.get('bbox')
                if not bbox:
                    continue
                
                x_center = (bbox[0] + bbox[2]) / 2
                col_idx = self._find_column_index(x_center, col_positions)
                
                key = f"{col_idx},{row_idx}"
                text = block.get('text', '').strip()
                
                if key in table_data:
                    table_data[key] += ' ' + text
                else:
                    table_data[key] = text
        
        return {
            'table_header': header_text,
            'table_data': table_data,
            'rows': len(rows),
            'cols': len(col_positions)
        }
    
    def _group_by_rows(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Group blocks into rows based on Y coordinate."""
        if not blocks:
            return []
        
        # Sort by Y coordinate
        sorted_blocks = sorted(blocks, key=lambda b: b.get('bbox', [0, 0, 0, 0])[1])
        
        rows = []
        current_row = []
        current_y = None
        
        for block in sorted_blocks:
            bbox = block.get('bbox')
            if not bbox:
                continue
            
            y = bbox[1]
            
            if current_y is None or abs(y - current_y) <= self.row_tolerance:
                current_row.append(block)
                if current_y is None:
                    current_y = y
            else:
                if current_row:
                    # Sort row by X coordinate
                    current_row.sort(key=lambda b: b.get('bbox', [0, 0, 0, 0])[0])
                    rows.append(current_row)
                current_row = [block]
                current_y = y
        
        if current_row:
            current_row.sort(key=lambda b: b.get('bbox', [0, 0, 0, 0])[0])
            rows.append(current_row)
        
        return rows
    
    def _find_column_positions(self, rows: List[List[Dict]]) -> List[int]:
        """Find column X positions from all rows."""
        all_x_positions = []
        
        for row in rows:
            for block in row:
                bbox = block.get('bbox')
                if bbox:
                    all_x_positions.append(bbox[0])  # Left edge
        
        if not all_x_positions:
            return []
        
        # Cluster X positions into columns
        all_x_positions.sort()
        columns = []
        current_col = all_x_positions[0]
        
        for x in all_x_positions:
            if abs(x - current_col) > self.col_tolerance:
                columns.append(current_col)
                current_col = x
        columns.append(current_col)
        
        return columns
    
    def _find_column_index(self, x: float, col_positions: List[int]) -> int:
        """Find which column a given X coordinate belongs to."""
        if not col_positions:
            return 0
        
        for i, col_x in enumerate(col_positions):
            if x < col_x + self.col_tolerance:
                return i
        
        return len(col_positions) - 1


class StructureAnalyzer:
    """
    Analyzes OCR text to detect document structure.
    Uses strict validation for section headers.
    """
    
    def __init__(self):
        self.hierarchy_tracker = SectionHierarchyTracker()
        self.table_detector = TableDetector()
    
    def classify_block(self, text: str, is_first_page: bool = False) -> Tuple[str, Optional[str], int]:
        """
        Classify a text block.
        
        Returns:
            (block_type, section_number, level)
        """
        text_stripped = text.strip()
        
        # Check for table headers first
        if self.table_detector.is_table_header(text_stripped):
            return ("table_header", None, 0)
        
        # Check for section headers with strict validation
        was_updated, section_num, level = self.hierarchy_tracker.update(text_stripped)
        if was_updated:
            block_type = "section_header" if level == 1 else "subsection_header"
            return (block_type, section_num, level)
        
        # Title detection (first page, short text, capitalized)
        if is_first_page and len(text_stripped.split()) <= 8 and len(text_stripped) > 3:
            words = text_stripped.split()
            if words and words[0][0].isupper():
                # Make sure it's not a common header/footer
                if not self._is_page_element(text_stripped):
                    return ("title", None, 0)
        
        # Page elements (headers/footers) - filter these out
        if self._is_page_element(text_stripped):
            return ("page_element", None, 0)
        
        return ("body", None, 0)
    
    def _is_page_element(self, text: str) -> bool:
        """Check if text is a page header/footer."""
        text_lower = text.lower().strip()
        
        # Page numbers
        if re.match(r'^page\s*\d+', text_lower):
            return True
        if re.match(r'^\d+\s*(of|de)\s*\d+$', text_lower):
            return True
        if re.match(r'^\d+$', text_lower) and len(text_lower) <= 4:
            return True
        
        # Common footer/header patterns
        footer_keywords = [
            'confidential', 'internal', 'uncontrolled', 'copyright', '©',
            'página', 'all rights reserved', 'proprietary', 'draft',
            'transmission', 'electricity networks'
        ]
        
        if len(text.split()) <= 10:
            text_check = text_lower
            if any(kw in text_check for kw in footer_keywords):
                return True
        
        return False
    
    def get_hierarchy(self) -> List[str]:
        return self.hierarchy_tracker.get_hierarchy()
    
    def get_current_section(self) -> Optional[str]:
        return self.hierarchy_tracker.get_current_section()
    
    def reset(self):
        self.hierarchy_tracker.reset()


class DocumentExtractor:
    """Extracts structured text from PDFs using PPOCRv5."""
    
    def __init__(self, dpi: int = DPI):
        self.ocr = None
        self.dpi = dpi
        self.structure_analyzer = StructureAnalyzer()
        self.table_detector = TableDetector()
        self.stats = {
            'pages': 0,
            'text_blocks': 0,
            'sections': 0,
            'tables': 0,
            'avg_confidence': 0.0
        }
    
    def init_ocr(self):
        """Initialize PPOCRv5 optimized for RTX GPU."""
        if self.ocr is not None:
            return
        
        logger.info("Initializing PPOCRv5...")
        try:
            self.ocr = PaddleOCR(
                lang='latin',
                ocr_version='PP-OCRv5',
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                text_det_limit_side_len=1920,
                text_det_limit_type='max',
                text_det_thresh=0.3,
                text_det_box_thresh=0.5,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.5,
                text_recognition_batch_size=16,
                use_doc_orientation_classify=False,
                use_textline_orientation=True,
                use_doc_unwarping=False,
            )
            logger.info("✓ PPOCRv5 initialized (server models, GPU)")
        except Exception as e:
            logger.warning(f"Server model failed: {e}")
            self.ocr = PaddleOCR(
                lang='latin',
                ocr_version='PP-OCRv5',
                use_textline_orientation=True,
            )
            logger.info("✓ PPOCRv5 initialized (default)")
    
    def cleanup(self):
        if self.ocr:
            del self.ocr
            self.ocr = None
            gc.collect()
    
    def _normalize_ocr_result(self, result: Any) -> List[Dict]:
        """
        Normalize PaddleOCR 3.3.2 result format.
        
        Returns list of: {'text': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}
        """
        normalized = []
        
        if not result:
            return normalized
        
        # Handle list wrapper
        data = result[0] if isinstance(result, list) and len(result) > 0 else result
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        
        # Extract from dict format (PaddleOCR 3.3.2)
        if isinstance(data, dict):
            rec_texts = data.get('rec_texts', [])
            rec_scores = data.get('rec_scores', [])
            rec_polys = data.get('rec_polys', [])
            
            for i, text in enumerate(rec_texts):
                if not text or not str(text).strip():
                    continue
                
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.5
                
                # Extract bbox from polygon
                bbox = None
                if i < len(rec_polys):
                    poly = rec_polys[i]
                    try:
                        xs = [int(p[0]) for p in poly]
                        ys = [int(p[1]) for p in poly]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    except:
                        pass
                
                normalized.append({
                    'text': str(text).strip(),
                    'confidence': conf,
                    'bbox': bbox
                })
        
        return normalized
    
    def process_page(self, img_bgr: np.ndarray, page_num: int,
                     is_first_page: bool = False) -> Tuple[List[TextBlock], List[Dict], float]:
        """
        Process a single page image with OCR.
        
        Returns:
            (blocks, table_data_list, avg_confidence)
        """
        try:
            result = self.ocr.predict(img_bgr)
        except Exception as e:
            logger.error(f"OCR failed on page {page_num}: {e}")
            return [], [], 0.0
        
        if not result:
            return [], [], 0.0
        
        ocr_data = self._normalize_ocr_result(result)
        
        if not ocr_data:
            return [], [], 0.0
        
        blocks = []
        tables = []
        confidences = []
        
        i = 0
        while i < len(ocr_data):
            item = ocr_data[i]
            text = item['text']
            conf = item['confidence']
            bbox = item.get('bbox')
            
            if not text.strip():
                i += 1
                continue
            
            # Check for table
            if self.table_detector.is_table_header(text):
                # Extract table structure
                end_idx, table_blocks = self.table_detector.detect_table_region(ocr_data, i)
                table_structure = self.table_detector.extract_table_structure(table_blocks, text)
                tables.append(table_structure)
                
                block = TextBlock(
                    text=text.strip(),
                    confidence=conf,
                    line_index=i,
                    block_type="table_header",
                    bbox=bbox
                )
                blocks.append(block)
                self.stats['tables'] += 1
                
                # Add table cells as body blocks (for text extraction)
                for tb in table_blocks:
                    blocks.append(TextBlock(
                        text=tb['text'],
                        confidence=tb['confidence'],
                        line_index=len(blocks),
                        block_type="table_cell",
                        bbox=tb.get('bbox')
                    ))
                
                i = end_idx
                continue
            
            # Classify the block
            block_type, section_num, level = self.structure_analyzer.classify_block(
                text, is_first_page
            )
            
            # Skip page elements
            if block_type == "page_element":
                i += 1
                continue
            
            block = TextBlock(
                text=text.strip(),
                confidence=conf,
                line_index=i,
                block_type=block_type,
                section_number=section_num,
                level=level,
                bbox=bbox
            )
            blocks.append(block)
            confidences.append(conf)
            
            self.stats['text_blocks'] += 1
            if block_type in ('section_header', 'subsection_header'):
                self.stats['sections'] += 1
            
            i += 1
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return blocks, tables, avg_conf
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict], Dict]:
        """Process entire PDF and return structured page data."""
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        
        self.structure_analyzer.reset()
        
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='jpeg',
                thread_count=2
            )
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return [], {'error': str(e)}
        
        pages_data = []
        total_conf = 0.0
        
        for page_num, pil_img in enumerate(tqdm(images, desc="OCR pages", leave=False), start=1):
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            is_first_page = (page_num == 1)
            blocks, tables, confidence = self.process_page(img_bgr, page_num, is_first_page)
            
            if blocks:
                pages_data.append({
                    'page_number': page_num,
                    'blocks': blocks,
                    'tables': tables,
                    'confidence': confidence,
                    'section_hierarchy': self.structure_analyzer.get_hierarchy(),
                    'current_section': self.structure_analyzer.get_current_section()
                })
                total_conf += confidence
            
            self.stats['pages'] += 1
            
            del img_rgb, img_bgr, pil_img
            gc.collect()
        
        del images
        gc.collect()
        
        self.stats['avg_confidence'] = total_conf / len(pages_data) if pages_data else 0.0
        
        return pages_data, {'total_pages': len(pages_data)}


def get_parent_section(section_num: Optional[str]) -> Optional[str]:
    """Get parent section number."""
    if not section_num:
        return None
    parts = section_num.rsplit('.', 1)
    return parts[0] if len(parts) > 1 else None


def create_chunks(
    pages_data: List[Dict],
    file_name: str,
    source_folder: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """Create document chunks from extracted page data."""
    chunks = []
    chunk_idx = 0
    
    # Track hierarchy independently during chunking
    current_hierarchy = []
    current_section_num = None
    
    # Track tables by page
    page_tables = {}
    for page_data in pages_data:
        page_num = page_data['page_number']
        page_tables[page_num] = page_data.get('tables', [])
    
    current_table_idx = {}  # Track which table we're processing per page
    
    for page_data in pages_data:
        page_num = page_data['page_number']
        blocks = page_data['blocks']
        tables = page_data.get('tables', [])
        
        current_text = []
        table_idx = 0
        in_table = False
        current_table_data = None
        
        for block in blocks:
            # Update hierarchy from section headers
            if block.section_number:
                current_section_num = block.section_number
                level = block.level if block.level else block.section_number.count('.') + 1
                current_hierarchy = current_hierarchy[:level-1]
                title = re.sub(r'^[\d.A-Z]+\s*', '', block.text, flags=re.IGNORECASE).strip()
                entry = f"{block.section_number} {title}".strip() if title else block.section_number
                current_hierarchy.append(entry)
            
            # Handle table headers
            if block.block_type == 'table_header':
                # Save accumulated body text first
                if current_text:
                    text = '\n'.join(current_text)
                    if text.strip():
                        chunks.append(DocumentChunk(
                            chunk_id=f"{file_name}__c{chunk_idx}",
                            file_name=file_name,
                            page_number=page_num,
                            chunk_index=chunk_idx,
                            chunk_text=text.strip(),
                            word_count=len(text.split()),
                            content_type="body",
                            section_hierarchy=current_hierarchy.copy(),
                            section_number=current_section_num,
                            parent_section=get_parent_section(current_section_num),
                            source_folder=source_folder
                        ))
                        chunk_idx += 1
                    current_text = []
                
                # Get table data if available
                if table_idx < len(tables):
                    current_table_data = tables[table_idx]
                    table_idx += 1
                else:
                    current_table_data = {'table_header': block.text, 'table_data': {}}
                
                in_table = True
                continue
            
            # Handle table cells
            if block.block_type == 'table_cell':
                # Skip - table data is captured in table_data structure
                continue
            
            # End table mode when we hit a non-table block
            if in_table and block.block_type not in ('table_header', 'table_cell'):
                # Create table chunk
                if current_table_data:
                    # Build table text representation
                    table_text = f"{current_table_data.get('table_header', 'Table')}\n"
                    td = current_table_data.get('table_data', {})
                    if td:
                        # Sort by row then column
                        sorted_keys = sorted(td.keys(), key=lambda k: (int(k.split(',')[1]), int(k.split(',')[0])))
                        current_row = -1
                        for key in sorted_keys:
                            col, row = map(int, key.split(','))
                            if row != current_row:
                                if current_row >= 0:
                                    table_text += '\n'
                                current_row = row
                            else:
                                table_text += ' | '
                            table_text += td[key]
                    
                    chunks.append(DocumentChunk(
                        chunk_id=f"{file_name}__c{chunk_idx}",
                        file_name=file_name,
                        page_number=page_num,
                        chunk_index=chunk_idx,
                        chunk_text=table_text.strip(),
                        word_count=len(table_text.split()),
                        content_type="table",
                        section_hierarchy=current_hierarchy.copy(),
                        section_number=current_section_num,
                        parent_section=get_parent_section(current_section_num),
                        has_table=True,
                        table_data=current_table_data,
                        source_folder=source_folder
                    ))
                    chunk_idx += 1
                
                in_table = False
                current_table_data = None
            
            # Headers get their own chunks
            if block.block_type in ('title', 'section_header', 'subsection_header'):
                # Save accumulated body text first
                if current_text:
                    text = '\n'.join(current_text)
                    if text.strip():
                        chunks.append(DocumentChunk(
                            chunk_id=f"{file_name}__c{chunk_idx}",
                            file_name=file_name,
                            page_number=page_num,
                            chunk_index=chunk_idx,
                            chunk_text=text.strip(),
                            word_count=len(text.split()),
                            content_type="body",
                            section_hierarchy=current_hierarchy.copy(),
                            section_number=current_section_num,
                            parent_section=get_parent_section(current_section_num),
                            source_folder=source_folder,
                            confidence=block.confidence
                        ))
                        chunk_idx += 1
                    current_text = []
                
                # Create header chunk
                chunks.append(DocumentChunk(
                    chunk_id=f"{file_name}__c{chunk_idx}",
                    file_name=file_name,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    chunk_text=block.text,
                    word_count=len(block.text.split()),
                    content_type=block.block_type,
                    section_hierarchy=current_hierarchy.copy(),
                    section_number=block.section_number,
                    parent_section=get_parent_section(block.section_number),
                    source_folder=source_folder,
                    confidence=block.confidence
                ))
                chunk_idx += 1
                
            elif block.block_type == 'body':
                # Accumulate body text
                current_text.append(block.text)
                
                # Check if we need to create a chunk
                combined = '\n'.join(current_text)
                if len(combined) >= chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{file_name}__c{chunk_idx}",
                        file_name=file_name,
                        page_number=page_num,
                        chunk_index=chunk_idx,
                        chunk_text=combined.strip(),
                        word_count=len(combined.split()),
                        content_type="body",
                        section_hierarchy=current_hierarchy.copy(),
                        section_number=current_section_num,
                        parent_section=get_parent_section(current_section_num),
                        source_folder=source_folder,
                        confidence=block.confidence
                    ))
                    chunk_idx += 1
                    
                    # Keep overlap
                    overlap_text = combined[-overlap:] if len(combined) > overlap else ""
                    current_text = [overlap_text] if overlap_text.strip() else []
        
        # End of page - save remaining content
        if in_table and current_table_data:
            # Save pending table
            table_text = f"{current_table_data.get('table_header', 'Table')}\n"
            td = current_table_data.get('table_data', {})
            if td:
                sorted_keys = sorted(td.keys(), key=lambda k: (int(k.split(',')[1]), int(k.split(',')[0])))
                current_row = -1
                for key in sorted_keys:
                    col, row = map(int, key.split(','))
                    if row != current_row:
                        if current_row >= 0:
                            table_text += '\n'
                        current_row = row
                    else:
                        table_text += ' | '
                    table_text += td[key]
            
            chunks.append(DocumentChunk(
                chunk_id=f"{file_name}__c{chunk_idx}",
                file_name=file_name,
                page_number=page_num,
                chunk_index=chunk_idx,
                chunk_text=table_text.strip(),
                word_count=len(table_text.split()),
                content_type="table",
                section_hierarchy=current_hierarchy.copy(),
                section_number=current_section_num,
                parent_section=get_parent_section(current_section_num),
                has_table=True,
                table_data=current_table_data,
                source_folder=source_folder
            ))
            chunk_idx += 1
        
        if current_text:
            text = '\n'.join(current_text)
            if text.strip():
                chunks.append(DocumentChunk(
                    chunk_id=f"{file_name}__c{chunk_idx}",
                    file_name=file_name,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    chunk_text=text.strip(),
                    word_count=len(text.split()),
                    content_type="body",
                    section_hierarchy=current_hierarchy.copy(),
                    section_number=current_section_num,
                    parent_section=get_parent_section(current_section_num),
                    source_folder=source_folder
                ))
                chunk_idx += 1
                current_text = []
    
    return chunks


def process_folder(
    folder_path: str,
    output_file: str,
    dpi: int = DPI,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> Dict:
    """Process all PDFs in a folder."""
    
    # Find PDFs
    pdf_files = []
    if os.path.isfile(folder_path) and folder_path.lower().endswith('.pdf'):
        pdf_files = [folder_path]
        folder_path = os.path.dirname(folder_path) or '.'
    else:
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, f))
    
    if not pdf_files:
        logger.error("No PDF files found")
        return {}
    
    logger.info(f"Found {len(pdf_files)} PDF(s)")
    
    # Initialize extractor
    extractor = DocumentExtractor(dpi=dpi)
    extractor.init_ocr()
    
    all_chunks = []
    failed_pdfs = []
    start_time = datetime.now()
    
    source_folder = os.path.basename(folder_path)
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            pages_data, meta = extractor.process_pdf(pdf_path)
            
            if 'error' in meta:
                failed_pdfs.append({'file': pdf_path, 'error': meta['error']})
                continue
            
            # Create chunks
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            safe_name = re.sub(r'[^\w\-]', '_', base_name.lower())
            
            chunks = create_chunks(
                pages_data,
                safe_name,
                source_folder,
                chunk_size,
                overlap
            )
            all_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            failed_pdfs.append({'file': pdf_path, 'error': str(e)})
        
        gc.collect()
    
    # Cleanup
    extractor.cleanup()
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Build output
    output = {
        'folder_path': os.path.abspath(folder_path),
        'extraction_date': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'total_chunks': len(all_chunks),
        'stats': extractor.stats,
        'failed_pdfs': failed_pdfs,
        'config': {
            'dpi': dpi,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'ocr_version': 'PP-OCRv5'
        },
        'chunks': [c.to_dict() for c in all_chunks]
    }
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Saved {len(all_chunks)} chunks to {output_file}")
    logger.info(f"  Processing time: {processing_time:.1f}s")
    logger.info(f"  Stats: {extractor.stats}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Extract structured text from PDFs using PPOCRv5'
    )
    parser.add_argument('input', help='PDF file or folder path')
    parser.add_argument('-o', '--output', default='ocr_output.json',
                        help='Output JSON file')
    parser.add_argument('--dpi', type=int, default=DPI,
                        help=f'DPI for PDF rendering (default: {DPI})')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                        help=f'Chunk size in characters (default: {CHUNK_SIZE})')
    parser.add_argument('--overlap', type=int, default=CHUNK_OVERLAP,
                        help=f'Chunk overlap in characters (default: {CHUNK_OVERLAP})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)
    
    process_folder(
        args.input,
        args.output,
        dpi=args.dpi,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )


if __name__ == '__main__':
    main()