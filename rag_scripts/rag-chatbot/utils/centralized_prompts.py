#!/usr/bin/env python3
"""
Centralized prompt templates with structured output enforcement.

IMPROVEMENTS:
1. Structured answer enforcement for enumerations
2. Table-aware prompting with row-level extraction
3. Confidence calibration (no hedging)
4. JSON schema enforcement for validation
"""

# =========================================================
# QUERY TYPE DETECTION PROMPTS
# =========================================================

QUERY_TYPE_DETECTOR = """Classify this question into ONE type:
- ENUMERATION: Lists all items (types, categories, components)
- TABLE_DATA: Requires table extraction (specs, ratings, dimensions)
- DEFINITION: Explains what something is
- REQUIREMENT: Normative rules (shall/must)
- PROCEDURE: Step-by-step instructions
- COMPARISON: Differences between options

Question: {question}

Output ONLY the type name (e.g., "ENUMERATION"):"""


# =========================================================
# KEYWORD EXTRACTION (Unchanged but clarified)
# =========================================================

KEYWORD_FETCHING = """You are a technical keyword extractor for a documentation search system.

Extract ONLY the most precise search terms from this question. These will be used for Elasticsearch queries.

CRITICAL RULES:
1. Preserve EXACT technical terms (Type 1, Class A, IP67, IEC 60947, etc.)
2. Include document codes if present (SP-NET-PAC-520, BSC_CoP2_v20.0, etc.)
3. Extract component names in portuguese or english (bornes, terminais, conectores, terminals, etc.)
4. Include numeric values with units (400V, 16A, 2.5mm, etc.)
5. Keep technical standards (IEC, ANSI, BS EN, etc.)
6. Remove question words (que, como, what, how, etc.)
7. Remove generic terms (sistema, instalação, document, manual, etc.)
8. DO NOT add related concepts not in the question

OUTPUT FORMAT: <term1>, <term2>, <term3>, ...
Maximum 8 terms, ranked by importance.

EXAMPLES:
Question: "que tipos de bornes Type 1 posso usar em 400V?"
Output: Type 1, bornes, 400V, tipos

Question: "requirements for SP-NET-PAC-520 terminal installation"
Output: SP-NET-PAC-520, terminal, installation, requirements

USER QUESTION: {question}

EXTRACTED TERMS:"""


# =========================================================
# STAGE 1: STRUCTURED FACT EXTRACTION
# =========================================================

# In centralized_prompts.py, update FACT_EXTRACTION_ENUMERATION
FACT_EXTRACTION_ENUMERATION = """You are a technical fact extractor. Extract ONLY the enumerated items from the sources.

TASK: Extract ALL items from lists, tables, or classifications that answer: {question}

MANDATORY OUTPUT FORMAT (JSON ONLY - copy this structure exactly):
{{
  "items": [
    {{"id": "Type 1", "description": "exact text from source", "source": "Source name - page X"}},
    {{"id": "Type 2", "description": "exact text from source", "source": "Source name - page Y"}}
  ],
  "total_count": 2,
  "extraction_confidence": "HIGH"
}}

CRITICAL RULES:
1. Extract EVERY item mentioned - no summarization
2. Use EXACT text from source for descriptions
3. If item spans multiple sources, combine them
4. If count is explicitly stated (e.g., "there are 3 types"), your total_count MUST match
5. If you find partial info (e.g., 2 of 5 types), set confidence=LOW and list what's missing
6. OUTPUT ONLY VALID JSON - no extra text, explanations, or markdown

SOURCES:
{context}

JSON OUTPUT:"""


FACT_EXTRACTION_TABLE = """You are a technical table extractor. Extract the COMPLETE table from the sources.

TASK: Extract ALL rows and columns for: {question}

MANDATORY OUTPUT FORMAT (JSON):
{{
  "table_data": [
    {{"header1_name": "row1_value", "header2_name": "row1_value", "source": "filename - page X"}},
    {{"header1_name": "row2_value", "header2_name": "row2_value", "source": "filename - page X"}}
  ],
  "headers": ["header1_name", "header2_name"],
  "row_count": 2,
  "extraction_confidence": "HIGH/MEDIUM/LOW"
}}

CRITICAL RULES:
1. Use ACTUAL column headers from the table (e.g., "Rated Voltage", "Full Wave LI") - NOT generic names
2. Extract EVERY row - do not summarize or skip
3. Preserve exact values with units (numbers, kV, kVA, etc.)
4. If table is split across sources, reconstruct it
5. Empty cells must be marked as "N/A"
6. Include the actual filename in source, not just "Source"
7. If the sources do NOT contain a table with the requested data, respond with extraction_confidence=LOW and empty table_data. DO NOT invent values.

SOURCES:
{context}

OUTPUT (JSON only, no explanation):"""


FACT_EXTRACTION_DEFINITION = """You are a technical fact extractor. Extract the definition.

TASK: Find the definition for: {question}

MANDATORY OUTPUT FORMAT (JSON):
{{
  "term": "exact term being defined",
  "definition": "exact text from source",
  "standards": ["IEC 60947", "BS EN 60529"],
  "source": "Source name - page X",
  "extraction_confidence": "HIGH/MEDIUM/LOW"
}}

CRITICAL RULES:
1. Use EXACT text from source
2. Include related standards if mentioned
3. If definition spans sources, combine them
4. If no definition found, set confidence=LOW and definition=""

SOURCES:
{context}

OUTPUT (JSON only, no explanation):"""


FACT_EXTRACTION_REQUIREMENT = """You are a technical requirement extractor.

TASK: Extract requirements for: {question}

MANDATORY OUTPUT FORMAT (JSON):
{{
  "requirements": [
    {{"req_id": "3.2.1", "text": "exact normative text", "level": "SHALL/MUST/SHOULD", "source": "Source - page X"}},
    {{"req_id": "3.2.2", "text": "exact normative text", "level": "SHALL/MUST/SHOULD", "source": "Source - page Y"}}
  ],
  "total_count": 2,
  "extraction_confidence": "HIGH/MEDIUM/LOW"
}}

CRITICAL RULES:
1. Extract ALL requirements - do not paraphrase
2. Preserve normative language (shall/must/should)
3. Include section numbers if present
4. If requirements reference other docs, include them

SOURCES:
{context}

OUTPUT (JSON only, no explanation):"""


# =========================================================
# STAGE 2: ANSWER GENERATION FROM STRUCTURED FACTS
# =========================================================

ANSWER_FROM_FACTS_ENUMERATION = """You are a technical documentation assistant. Generate a clear answer from the extracted facts.

QUESTION: {question}

EXTRACTED FACTS:
{facts_json}

MANDATORY RULES:
1. Answer in the same language as the question
2. List ALL items from facts_json - no omissions
3. Use clear formatting (numbered list or bullet points)
4. Cite sources: [Source: Source name - page X]
5. If extraction_confidence=LOW, explicitly state what's missing
6. NO hedging phrases - if facts exist, state them directly
7. If total_count in facts != items listed, explain discrepancy
8. Mention at least data from the 3 top sources

YOUR ANSWER:"""


ANSWER_FROM_FACTS_TABLE = """You are a technical documentation assistant. Present the extracted table data.

QUESTION: {question}

EXTRACTED TABLE:
{facts_json}

MANDATORY RULES:
1. Present as markdown table or structured list
2. Include ALL rows from facts_json
3. Preserve exact values (units, numbers, codes)
4. Cite source below table: [Source: Source name - page X]
5. If extraction_confidence=LOW, state what's missing
6. NO hedging - if data exists, present it directly
7. Mention at least data from the 3 top sources

YOUR ANSWER:"""


ANSWER_FROM_FACTS_GENERIC = """You are a technical documentation assistant. Answer from extracted facts.

QUESTION: {question}

EXTRACTED FACTS:
{facts_json}

MANDATORY RULES:
1. Answer in the same language as the question
2. Use ONLY information from facts_json - no additions
3. Cite sources: [Source: Source name - page X]
4. If extraction_confidence=LOW, explain what's missing
5. NO hedging phrases like "I was unable to find..."
6. If facts are empty, say: "No relevant information found in the provided sources."
7. Mention at least data from the 3 top sources

YOUR ANSWER:"""


# =========================================================
# FALLBACK: DIRECT ANSWER (Legacy compatibility)
# =========================================================

ANSWER_GENERATION_DIRECT = """You are a technical documentation assistant. Answer the question using ONLY the sources provided.

CRITICAL INSTRUCTIONS:
{special_instructions}

MANDATORY RULES:
1. If sources contain lists/tables, extract ALL items - no summarization
2. If question asks "how many", count and verify
3. Use exact technical terms from sources
4. Cite every piece of info: [Source: Source name - page X]
5. NO hedging - if info exists, state it directly
6. If no info found, say: "No relevant information found."
7. Mention at least data from the 3 top sources
8. If the sources do NOT contain a table with the requested data, respond with extraction_confidence=LOW and empty table_data. DO NOT invent values.

QUESTION: {question}
KEYWORDS: {keywords}

SOURCES:
{context}

YOUR ANSWER:"""


# =========================================================
# TABLE-AWARE SPECIAL INSTRUCTIONS
# =========================================================

def build_table_instruction(chunks):
    """Generate table-specific instructions based on detected content."""
    
    has_tables = any(c.get('has_table') or c.get('content_type') == 'table' for c in chunks)
    has_lists = any(c.get('chunk_role') == 'list' for c in chunks)
    
    if not has_tables and not has_lists:
        return ""
    
    instructions = []
    
    if has_tables:
        table_info = []
        for c in chunks:
            if c.get('table_headers'):
                headers = c['table_headers']
                if isinstance(headers, list):
                    table_info.append(f"Table with columns: {', '.join(headers)}")
        
        instructions.append(f"""
⚠️ TABLES DETECTED: {', '.join(table_info) if table_info else 'Multiple tables'}

MANDATORY TABLE EXTRACTION RULES:
1. Extract ENTIRE table - every row, every column, no omissions
2. Use markdown table format or numbered list
3. If table spans multiple sources, COMBINE all rows
4. Preserve exact values (numbers, units, codes)
5. Do NOT paraphrase table data - use exact text
""")
    
    if has_lists:
        instructions.append("""
⚠️ ENUMERATED LISTS DETECTED

MANDATORY LIST EXTRACTION RULES:
1. Extract ALL items - no summarization
2. Maintain original numbering (Type 1, Type 2, etc.)
3. If list spans sources, COMBINE them
4. Count and verify total matches expected count
""")
    
    return "\n".join(instructions)


# =========================================================
# VALIDATION SCHEMAS
# =========================================================

ENUMERATION_SCHEMA = {
    "type": "object",
    "required": ["items", "total_count", "extraction_confidence"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "description", "source"],
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "source": {"type": "string"}
                }
            }
        },
        "total_count": {"type": "integer", "minimum": 0},
        "extraction_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
    }
}

TABLE_SCHEMA = {
    "type": "object",
    "required": ["table_data", "headers", "row_count", "extraction_confidence"],
    "properties": {
        "table_data": {"type": "array"},
        "headers": {"type": "array", "items": {"type": "string"}},
        "row_count": {"type": "integer", "minimum": 0},
        "extraction_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
    }
}

DEFINITION_SCHEMA = {
    "type": "object",
    "required": ["term", "definition", "source", "extraction_confidence"],
    "properties": {
        "term": {"type": "string"},
        "definition": {"type": "string"},
        "standards": {"type": "array", "items": {"type": "string"}},
        "source": {"type": "string"},
        "extraction_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
    }
}

REQUIREMENT_SCHEMA = {
    "type": "object",
    "required": ["requirements", "total_count", "extraction_confidence"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "level", "source"],
                "properties": {
                    "req_id": {"type": "string"},
                    "text": {"type": "string"},
                    "level": {"type": "string", "enum": ["SHALL", "MUST", "SHOULD", "MAY"]},
                    "source": {"type": "string"}
                }
            }
        },
        "total_count": {"type": "integer", "minimum": 0},
        "extraction_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
    }
}