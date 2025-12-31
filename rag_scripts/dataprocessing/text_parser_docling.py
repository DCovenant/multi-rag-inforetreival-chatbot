#!/usr/bin/env python3
"""
RAG Document Extraction with Docling
====================================

Simple, clean extraction using IBM's Docling library.

Requirements:
  pip install docling tqdm

Usage:
  python indexer-docling.py pdfs/ -o output.json
"""
import os
import sys
import argparse
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
# Initialize Docling with parallel processing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from tqdm import tqdm

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# At module level
SECTION_PATTERN = re.compile(r'\b(\d{1,2})\.(\d{1,2})(?:\.\d{1,2})*\b')

@dataclass 
class DocumentChunk:
    """A chunk ready for Elasticsearch indexing."""
    chunk_id: str
    file_name: str
    page_number: int
    chunk_index: int
    chunk_text: str
    word_count: int
    content_type: str  # body, table, section_header, title
    parent_sections: List[str] = field(default_factory=list)  # ["1", "2"] for chunks with 1.x and 2.x
    has_table: bool = False
    table_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class SectionTracker:
    PATTERNS = [
        (re.compile(r'^(\d{1,2})\s+([A-Z][a-zA-Z\s]{2,50})$', re.IGNORECASE), 1),
        (re.compile(r'^(\d{1,2}\.\d{1,2})\s+([A-Z][a-zA-Z\s]{2,60})$', re.IGNORECASE), 2),
        (re.compile(r'^(\d{1,2}\.\d{1,2}\.\d{1,2})\s+(.{2,60})$', re.IGNORECASE), 3),
        (re.compile(r'^(Appendix\s+[A-Z])\s*[-:]?\s*(.{0,60})$', re.IGNORECASE), 1),
    ]
    
    def __init__(self):
        self.hierarchy = []
        self.current = None
    
    def update(self, text: str) -> Tuple[bool, Optional[str], int]:
        """Try to parse text as section header. Returns (is_section, number, level)."""
        text = text.strip()
        if len(text) < 3 or len(text) > 80:
            return False, None, 0
        
        for pattern, level in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1).strip()
                title = match.group(2).strip() if match.lastindex >= 2 else ''
                
                self.hierarchy = self.hierarchy[:level-1]
                self.hierarchy.append(f"{num} {title}".strip())
                self.current = num
                return True, num, level
        
        return False, None, 0
    
    def reset(self):
        self.hierarchy = []
        self.current = None

def extract_parent_sections(text: str) -> List[str]:
    matches = SECTION_PATTERN.findall(text)
    parents = sorted(set(m[0] for m in matches), key=int)
    return parents

def get_parent_section(section_num: Optional[str]) -> Optional[str]:
    """Get parent section: '1.2.3' -> '1.2'"""
    if not section_num:
        return None
    parts = section_num.rsplit('.', 1)
    return parts[0] if len(parts) > 1 else None


def extract_table_cells(table_item) -> Dict:
    """Extract table cell data from Docling table item."""
    table_data = {'table_header': '', 'table_data': {}, 'rows': 0, 'cols': 0}
    
    if not hasattr(table_item, 'data') or not table_item.data:
        return table_data
    
    data = table_item.data
    if not hasattr(data, 'table_cells'):
        return table_data
    
    for cell in data.table_cells:
        row = cell.start_row_offset_idx
        col = cell.start_col_offset_idx
        text = cell.text or ''
        
        table_data['table_data'][f"{col},{row}"] = text
        table_data['rows'] = max(table_data['rows'], row + 1)
        table_data['cols'] = max(table_data['cols'], col + 1)
    
    return table_data


def process_pdf(converter: DocumentConverter, pdf_path: str) -> Tuple[List[Dict], Dict]:
    """
    Process PDF with Docling.
    Returns (pages_data, stats)
    """
    result = converter.convert(pdf_path)
    doc = result.document
    
    tracker = SectionTracker()
    stats = {'pages': 0, 'blocks': 0, 'tables': 0, 'sections': 0}
    
    # Group blocks by page
    pages = {}  # page_num -> list of blocks
    
    for item, _ in doc.iterate_items():
        # Get page number from provenance
        page_num = 1
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, 'page_no') and prov.page_no is not None:
                page_num = int(prov.page_no) + 1  # 0-indexed -> 1-indexed
        
        # Get text content
        text = ''
        item_type = type(item).__name__.lower()
        
        if 'table' in item_type:
            # Tables: use export_to_markdown for text representation
            try:
                text = item.export_to_markdown(doc)
            except:
                text = str(item.data) if hasattr(item, 'data') else ''
            
            block = {
                'text': text.strip(),
                'block_type': 'table',
                'table_data': extract_table_cells(item),
                'section_number': tracker.current,
                'level': 0
            }
            stats['tables'] += 1
        else:
            # Text items: get text attribute
            if hasattr(item, 'text') and item.text:
                text = item.text.strip()
            
            if not text:
                continue
            
            # Check if it's a section header
            is_section, section_num, level = tracker.update(text)
            
            if is_section:
                block_type = 'section_header' if level == 1 else 'subsection_header'
                stats['sections'] += 1
            elif 'heading' in item_type or 'title' in item_type:
                block_type = 'title'
            else:
                block_type = 'body'
            
            block = {
                'text': text,
                'block_type': block_type,
                'section_number': section_num if is_section else None,
                'level': level
            }
        
        # Add to page
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(block)
        stats['blocks'] += 1
    
    # Convert to sorted list
    pages_data = []
    for page_num in sorted(pages.keys()):
        pages_data.append({
            'page_number': page_num,
            'blocks': pages[page_num],
            'section_hierarchy': tracker.hierarchy.copy(),
            'current_section': tracker.current
        })
    
    stats['pages'] = len(pages_data)
    return pages_data, stats


def create_chunks(
    pages_data: List[Dict],
    file_name: str,
    chunk_size: int,
    overlap: int
) -> List[DocumentChunk]:
    """Create document chunks from pages data."""
    chunks = []
    chunk_idx = 0
    
    current_text = []
    current_page = 1
    
    for page_data in pages_data:
        page_num = page_data['page_number']
        
        for block in page_data['blocks']:
            text = block.get('text', '').strip()
            if not text:
                continue
            
            block_type = block.get('block_type', 'body')
            table_data = block.get('table_data')
            
            # Tables and headers get their own chunks
            if block_type in ('table', 'section_header', 'subsection_header', 'title'):
                # Flush accumulated body text first
                if current_text:
                    combined = '\n'.join(current_text)
                    if len(combined.strip()) > 20:
                        chunks.append(DocumentChunk(
                            chunk_id=f"{file_name}__c{chunk_idx}",
                            file_name=file_name,
                            page_number=current_page,
                            chunk_index=chunk_idx,
                            chunk_text=combined.strip(),
                            word_count=len(combined.split()),
                            content_type="body",
                            parent_sections=extract_parent_sections(combined)
                        ))
                        chunk_idx += 1
                    current_text = []
                
                # Create chunk for this block
                chunks.append(DocumentChunk(
                    chunk_id=f"{file_name}__c{chunk_idx}",
                    file_name=file_name,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    chunk_text=text,
                    word_count=len(text.split()),
                    content_type=block_type,
                    parent_sections=extract_parent_sections(text),
                    has_table=(block_type == 'table'),
                    table_data=table_data
                ))
                chunk_idx += 1
            else:
                # Accumulate body text
                current_text.append(text)
                current_page = page_num
                
                # Check if chunk is full
                combined = '\n'.join(current_text)
                if len(combined) >= chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{file_name}__c{chunk_idx}",
                        file_name=file_name,
                        page_number=current_page,
                        chunk_index=chunk_idx,
                        chunk_text=combined.strip(),
                        word_count=len(combined.split()),
                        content_type="body",
                        parent_sections=extract_parent_sections(combined)
                    ))
                    chunk_idx += 1
                    
                    # Keep overlap
                    current_text = [combined[-overlap:]] if len(combined) > overlap else []
    
    # Flush remaining text
    if current_text:
        combined = '\n'.join(current_text)
        if len(combined.strip()) > 20:
            chunks.append(DocumentChunk(
                chunk_id=f"{file_name}__c{chunk_idx}",
                file_name=file_name,
                page_number=current_page,
                chunk_index=chunk_idx,
                chunk_text=combined.strip(),
                word_count=len(combined.split()),
                content_type="body",
                parent_sections=extract_parent_sections(combined)
            ))
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDFs using Docling')
    parser.add_argument('input', help='PDF file or folder')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--overlap', type=int, default=CHUNK_OVERLAP)
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()
    
    # Find PDFs
    if os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        pdf_files = [args.input]
        folder_path = os.path.dirname(args.input) or '.'
    else:
        folder_path = args.input
        pdf_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(folder_path)
            for f in files if f.lower().endswith('.pdf')
        ]
    
    if not pdf_files:
        print("No PDF files found")
        sys.exit(1)
    
    folder_name = os.path.basename(os.path.abspath(folder_path))
    output_file = re.sub(r'[^\w\-]', '_', folder_name.lower()) + '.json'
    
    print(f"Found {len(pdf_files)} PDF(s), using {args.workers} workers")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Batch convert all PDFs at once
    print("Converting PDFs...")
    start_time = datetime.now()
    results = converter.convert_all(pdf_files, raises_on_error=False)
    
    all_chunks = []
    all_stats = {'pages': 0, 'blocks': 0, 'tables': 0, 'sections': 0}
    failed = []
    
    for result in tqdm(results, desc="Processing", total=len(pdf_files)):
        try:
            if result.status.name != "SUCCESS":
                failed.append({'file': result.input.file, 'error': str(result.status)})
                continue
            
            doc = result.document
            pdf_path = str(result.input.file)
            
            # Process document
            tracker = SectionTracker()
            stats = {'pages': 0, 'blocks': 0, 'tables': 0, 'sections': 0}
            pages = {}
            
            for item, _ in doc.iterate_items():
                page_num = 1
                if hasattr(item, 'prov') and item.prov:
                    prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                    if hasattr(prov, 'page_no') and prov.page_no is not None:
                        page_num = int(prov.page_no) + 1
                
                text = ''
                item_type = type(item).__name__.lower()
                
                if 'table' in item_type:
                    try:
                        text = item.export_to_markdown(doc)
                    except:
                        text = ''
                    
                    block = {
                        'text': text.strip(),
                        'block_type': 'table',
                        'table_data': extract_table_cells(item),
                        'section_number': tracker.current,
                        'level': 0
                    }
                    stats['tables'] += 1
                else:
                    if hasattr(item, 'text') and item.text:
                        text = item.text.strip()
                    
                    if not text:
                        continue
                    
                    is_section, section_num, level = tracker.update(text)
                    
                    if is_section:
                        block_type = 'section_header' if level == 1 else 'subsection_header'
                        stats['sections'] += 1
                    elif 'heading' in item_type or 'title' in item_type:
                        block_type = 'title'
                    else:
                        block_type = 'body'
                    
                    block = {
                        'text': text,
                        'block_type': block_type,
                        'section_number': section_num if is_section else None,
                        'level': level
                    }
                
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(block)
                stats['blocks'] += 1
            
            pages_data = [
                {'page_number': p, 'blocks': pages[p]}
                for p in sorted(pages.keys())
            ]
            stats['pages'] = len(pages_data)
            
            for k in all_stats:
                all_stats[k] += stats[k]
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            safe_name = re.sub(r'[^\w\-]', '_', base_name.lower())
            
            chunks = create_chunks(pages_data, safe_name, args.chunk_size, args.overlap)
            all_chunks.extend(chunks)
            
        except Exception as e:
            failed.append({'file': str(result.input.file), 'error': str(e)})
    
    # Write output
    output = {
        'folder_path': os.path.abspath(folder_path),
        'extraction_date': datetime.now().isoformat(),
        'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
        'total_chunks': len(all_chunks),
        'stats': all_stats,
        'failed_pdfs': failed,
        'config': {
            'chunk_size': args.chunk_size,
            'overlap': args.overlap,
            'extractor': 'docling'
        },
        'chunks': [c.to_dict() for c in all_chunks]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved {len(all_chunks)} chunks to {output_file}")
    print(f"  Stats: {all_stats}")

if __name__ == '__main__':
    main()