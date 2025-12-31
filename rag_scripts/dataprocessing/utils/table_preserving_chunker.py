#!/usr/bin/env python3
"""
Table-Preserving Smart Chunking Module

IMPROVEMENTS:
1. Tables never split across chunks
2. Header-row-column structure preserved
3. Context-aware boundaries (respects paragraphs, sections, tables)
4. Memory-safe (streaming compatible)
"""

import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# =========================================================
# TABLE DETECTION AND BOUNDARY MARKING
# =========================================================

def detect_table_boundaries(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect tables in text and return their boundaries.
    
    Returns:
        List of (start_pos, end_pos, table_type)
        table_type: "markdown", "ascii", "extracted"
    """
    boundaries = []
    
    # Pattern 1: Markdown tables (|---|---|)
    markdown_pattern = r'(\|[^\n]+\|[\n\r]+\|[-\s|]+\|[\n\r]+(?:\|[^\n]+\|[\n\r]+)+)'
    for match in re.finditer(markdown_pattern, text, re.MULTILINE):
        boundaries.append((match.start(), match.end(), "markdown"))
    
    # Pattern 2: ASCII tables (detected by consistent spacing/alignment)
    # Look for 3+ lines with consistent column-like structure
    ascii_pattern = r'((?:^.{10,}[\n\r]+){3,})'  # 3+ lines with content
    for match in re.finditer(ascii_pattern, text, re.MULTILINE):
        candidate = match.group(0)
        # Verify it's actually table-like (consistent spacing)
        if _is_ascii_table(candidate):
            boundaries.append((match.start(), match.end(), "ascii"))
    
    # Pattern 3: Extracted tables (marked during pdfplumber extraction)
    extracted_pattern = r'\[Table with \d+ rows x \d+ columns\].*?(?=\n\n|\Z)'
    for match in re.finditer(extracted_pattern, text, re.DOTALL):
        boundaries.append((match.start(), match.end(), "extracted"))
    
    # Sort by position and merge overlapping
    boundaries.sort(key=lambda x: x[0])
    merged = _merge_overlapping_boundaries(boundaries)
    
    logger.info(f"Detected {len(merged)} tables in text")
    return merged


def _is_ascii_table(text: str) -> bool:
    """Heuristic to detect ASCII table structure."""
    lines = text.split('\n')
    if len(lines) < 3:
        return False
    
    # Check for consistent whitespace patterns
    # (tables have columns aligned with spaces)
    space_positions = []
    for line in lines[:5]:  # Check first 5 lines
        spaces = [m.start() for m in re.finditer(r'\s{2,}', line)]
        space_positions.append(set(spaces))
    
    # If 3+ lines share common space positions, likely a table
    if len(space_positions) < 3:
        return False
    
    common_spaces = set.intersection(*space_positions)
    return len(common_spaces) >= 2  # At least 2 column gaps


def _merge_overlapping_boundaries(boundaries: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Merge overlapping table boundaries."""
    if not boundaries:
        return []
    
    merged = [boundaries[0]]
    for start, end, ttype in boundaries[1:]:
        prev_start, prev_end, prev_type = merged[-1]
        
        # If overlapping, merge
        if start <= prev_end:
            merged[-1] = (prev_start, max(end, prev_end), prev_type)
        else:
            merged.append((start, end, ttype))
    
    return merged


# =========================================================
# TABLE-PRESERVING CHUNKING
# =========================================================

def chunk_with_table_preservation(
    text: str,
    page_num: int,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict]:
    """
    Chunk text while preserving table boundaries.
    
    RULES:
    1. Tables < max_table_size: kept whole in single chunk
    2. Tables > max_table_size: get dedicated chunk, may exceed limit
    3. No table ever split across chunks
    4. Overlaps avoid splitting tables
    
    Args:
        text: Full page text
        page_num: Page number
        chunk_size: Target chunk size (soft limit)
        overlap: Overlap between chunks (soft limit)
        max_table_size: Max table size before forcing dedicated chunk
    
    Returns:
        List of chunk dicts with table metadata
    """
    
    # Detect table boundaries
    table_boundaries = detect_table_boundaries(text)
    
    if not table_boundaries:
        # No tables, use standard chunking
        return _chunk_segment(text, page_num, chunk_size, overlap, start_index=0, offset=0)
    
    # Build chunks with table awareness
    chunks = []
    current_pos = 0
    chunk_index = 0
    
    while current_pos < len(text):
        # Check if we're at/near a table
        table_info = _get_table_at_position(current_pos, table_boundaries)
        
        if table_info:
            start, end, ttype = table_info
            table_text = text[start:end]
            table_size = end - start
            
            # CASE 1: Small table - try to include with surrounding context
            if table_size < chunk_size:
                # Get context before table
                context_start = max(0, start - overlap)
                # Align to paragraph boundary
                context_start = _align_to_paragraph(text, context_start, forward=True)
                
                # Get context after table
                context_end = min(len(text), end + (chunk_size - table_size) // 2)
                context_end = _align_to_paragraph(text, context_end, forward=False)
                
                chunk_text = text[context_start:context_end]
                
                chunks.append({
                    "chunk_text": chunk_text.strip(),
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "has_table": True,
                    "table_type": ttype,
                    "start_pos": context_start,
                    "end_pos": context_end
                })
                
                chunk_index += 1
                current_pos = context_end
            
            # CASE 2: Large table - dedicated chunk (may exceed chunk_size)
            else:
                logger.warning(f"Large table ({table_size} chars) at page {page_num}, creating dedicated chunk")
                
                chunks.append({
                    "chunk_text": table_text.strip(),
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "has_table": True,
                    "table_type": ttype,
                    "is_large_table": True,
                    "start_pos": start,
                    "end_pos": end
                })
                
                chunk_index += 1
                current_pos = end
        
        else:
            # CASE 3: Regular text (no table nearby)
            # Standard chunking until next table or end
            next_table_pos = _get_next_table_position(current_pos, table_boundaries)
            
            if next_table_pos is None:
                # No more tables, chunk to end
                end_pos = len(text)
            else:
                # Chunk up to table boundary
                end_pos = next_table_pos
            
            # Standard chunking of this segment
            segment_chunks = _chunk_segment(
                text[current_pos:end_pos],
                page_num,
                chunk_size,
                overlap,
                start_index=chunk_index,
                offset=current_pos
            )
            
            chunks.extend(segment_chunks)
            chunk_index += len(segment_chunks)
            current_pos = end_pos
    
    logger.info(f"Page {page_num}: {len(chunks)} chunks (table-aware)")
    return chunks


def _get_table_at_position(pos: int, boundaries: List[Tuple[int, int, str]]) -> Optional[Tuple[int, int, str]]:
    """Check if position is inside a table."""
    for start, end, ttype in boundaries:
        if start <= pos < end:
            return (start, end, ttype)
    return None


def _get_next_table_position(pos: int, boundaries: List[Tuple[int, int, str]]) -> Optional[int]:
    """Get position of next table after pos."""
    for start, end, ttype in boundaries:
        if start > pos:
            return start
    return None


def _align_to_paragraph(text: str, pos: int, forward: bool = True) -> int:
    """Align position to nearest paragraph boundary."""
    if forward:
        # Find next paragraph break
        match = re.search(r'\n\n', text[pos:pos+200])
        if match:
            return pos + match.end()
    else:
        # Find previous paragraph break
        match = re.search(r'\n\n', text[max(0, pos-200):pos])
        if match:
            return max(0, pos - 200) + match.start()
    
    return pos

def _chunk_segment(
    text: str,
    page_num: int,
    chunk_size: int,
    overlap: int,
    start_index: int = 0,
    offset: int = 0
) -> List[Dict]:
    """Standard chunking for text segments without tables."""
    chunks = []
    
    # Split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return []
    
    current_chunk = []
    current_size = 0
    chunk_idx = start_index
    
    for para in paragraphs:
        para_len = len(para)
        
        # If adding this para exceeds chunk_size
        if current_size + para_len > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "chunk_id":chunk_idx,
                "file_name":_,
                "page_number": page_num,
                "chunk_index":_,
                "chunk_text": chunk_text.strip(),
                "word_count":_,
                "content_type":_,
                "chunk_role":_,
                "has_table": False,
                "start_pos": offset,
                "end_pos": offset + len(chunk_text)
            })
            
            # Start new chunk with overlap
            chunk_idx += 1
            overlap_paras = _get_overlap_paragraphs(current_chunk, overlap)
            current_chunk = overlap_paras
            current_size = sum(len(p) for p in current_chunk)
        
        current_chunk.append(para)
        current_size += para_len
    
    # Save remaining chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append({
            "chunk_text": chunk_text.strip(),
            "page_number": page_num,
            "chunk_index": chunk_idx,
            "has_table": False,
            "start_pos": offset,
            "end_pos": offset + len(chunk_text)
        })
    
    return chunks


def _get_overlap_paragraphs(paragraphs: List[str], overlap_size: int) -> List[str]:
    """Get last N paragraphs that fit in overlap size."""
    overlap_paras = []
    overlap_len = 0
    
    for para in reversed(paragraphs):
        if overlap_len + len(para) <= overlap_size:
            overlap_paras.insert(0, para)
            overlap_len += len(para)
        else:
            break
    
    return overlap_paras
