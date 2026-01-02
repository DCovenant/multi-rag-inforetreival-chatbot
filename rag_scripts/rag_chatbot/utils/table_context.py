"""
Table-aware context builder for LLM consumption.
Handles coordinate-based table_data format: {"0,0": "val", "1,0": "val", ...}
"""

from typing import List, Dict


def build_context(chunks: List[Dict], max_chars: int = 24000) -> str:
    """Build context string, rendering tables as markdown."""
    parts = []
    chars = 0
    
    for i, c in enumerate(chunks, 1):
        header = f"[Source {i}: {c.get('file_name', '?')} - Page {c.get('page_number', '?')}]"
        
        # Render table or text
        if c.get('content_type') == 'table' and c.get('table_data'):
            content = _render_table(c['table_data'])
            if c.get('table_caption'):
                content = f"Caption: {c['table_caption']}\n{content}"
        else:
            content = c.get('chunk_text', '')
        
        chunk = f"{header}\n{content}"
        
        if chars + len(chunk) > max_chars:
            break
        
        parts.append(chunk)
        chars += len(chunk)
    
    return "\n\n---\n\n".join(parts)


def _render_table(table_data) -> str:
    """
    Render table as markdown.
    
    Handles two formats:
    1. Coordinate format: {"table_data": {"0,0": "val", "1,0": "val"}, "rows": 5, "cols": 6}
    2. List format: [{"col1": "val1", "col2": "val2"}, ...]
    """
    if not table_data:
        return "[Empty table]"
    
    # Check if it's the coordinate format (nested dict with "table_data" key)
    if isinstance(table_data, dict) and "table_data" in table_data:
        return _render_coordinate_table(table_data)
    
    # List of dicts format
    if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
        return _render_list_table(table_data)
    
    # Fallback
    return str(table_data)


def _render_coordinate_table(table_info: Dict) -> str:
    """
    Render coordinate-based table: {"table_data": {"col,row": "value"}, "rows": N, "cols": M}
    Format: "col,row" where col=column index, row=row index (0=header row)
    """
    data = table_info.get("table_data", {})
    rows = table_info.get("rows", 0)
    cols = table_info.get("cols", 0)
    
    if not data or rows == 0 or cols == 0:
        return "[Empty table]"
    
    # Build grid: grid[row][col]
    grid = []
    for row_idx in range(rows):
        row = []
        for col_idx in range(cols):
            key = f"{col_idx},{row_idx}"
            row.append(str(data.get(key, "")))
        grid.append(row)
    
    # First row is headers
    headers = grid[0] if grid else []
    data_rows = grid[1:] if len(grid) > 1 else []
    
    # Build markdown table
    lines = []
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def _render_list_table(table_data: List[Dict]) -> str:
    """Render list-of-dicts table format."""
    # Infer headers from first row
    headers = [k for k in table_data[0].keys() if k != 'source']
    
    if not headers:
        return "\n".join(str(row) for row in table_data)
    
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|"
    ]
    
    for row in table_data:
        vals = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    
    return "\n".join(lines)