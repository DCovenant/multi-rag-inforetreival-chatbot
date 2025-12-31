#!/usr/bin/env python3
"""
Two-Stage Answer Generation with Structured Output Validation

STAGE 1: Extract structured facts (JSON)
STAGE 2: Generate natural answer from facts (no new info allowed)

Includes:
- Query type detection
- Schema validation
- Auto-retry on validation failure
- Confidence calibration
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from jsonschema import validate, ValidationError
import requests

logger = logging.getLogger(__name__)

# Import prompts (assumes centralized_prompts.py is available)
try:
    from utils.centralized_prompts import (
        QUERY_TYPE_DETECTOR,
        FACT_EXTRACTION_ENUMERATION,
        FACT_EXTRACTION_TABLE,
        FACT_EXTRACTION_DEFINITION,
        FACT_EXTRACTION_REQUIREMENT,
        ANSWER_FROM_FACTS_ENUMERATION,
        ANSWER_FROM_FACTS_TABLE,
        ANSWER_FROM_FACTS_GENERIC,
        ENUMERATION_SCHEMA,
        TABLE_SCHEMA,
        DEFINITION_SCHEMA,
        REQUIREMENT_SCHEMA,
        build_table_instruction
    )
except ImportError:
    logger.error("Failed to import prompts. Ensure centralized_prompts.py is in utils/")
    raise

DEBUGS = False

# =========================================================
# QUERY TYPE DETECTION
# =========================================================

class QueryType:
    """Query classification for pipeline routing."""
    ENUMERATION = "ENUMERATION"
    TABLE_DATA = "TABLE_DATA"
    DEFINITION = "DEFINITION"
    REQUIREMENT = "REQUIREMENT"
    PROCEDURE = "PROCEDURE"
    COMPARISON = "COMPARISON"
    GENERIC = "GENERIC"


def detect_query_type(question: str, ask_ollama_fn) -> str:
    """
    Detect query type using LLM classification.
    
    Falls back to heuristics if LLM fails.
    """
    try:
        prompt = QUERY_TYPE_DETECTOR.format(question=question)
        response = ask_ollama_fn(prompt, temperature=0.0, timeout=10)
        
        # Parse response
        response_upper = response.strip().upper()
        for qtype in [
            QueryType.ENUMERATION,
            QueryType.TABLE_DATA,
            QueryType.DEFINITION,
            QueryType.REQUIREMENT,
            QueryType.PROCEDURE,
            QueryType.COMPARISON
        ]:
            if qtype in response_upper:
                logger.info(f"Detected query type: {qtype}")
                return qtype
        
        # Fallback to heuristic
        return _heuristic_query_type(question)
        
    except Exception as e:
        logger.warning(f"Query type detection failed: {e}, using heuristics")
        return _heuristic_query_type(question)


def _heuristic_query_type(question: str) -> str:
    """Heuristic fallback for query type detection."""
    logger.info("Using heuristic query type fallback.")
    q_lower = question.lower()
    
    # Enumeration patterns
    enum_patterns = [
        r'\bhow many\b', r'\blist\b', r'\btypes?\b', r'\bkinds?\b',
        r'\bquais\b', r'\bquantos\b', r'\btipos?\b',
        r'\bname.*all\b', r'\benumerate\b'
    ]
    if any(re.search(p, q_lower) for p in enum_patterns):
        return QueryType.ENUMERATION
    
    # Table data patterns
    if any(x in q_lower for x in ['table', 'tabela', 'rating', 'specification', 'dimension']):
        return QueryType.TABLE_DATA
    
    # Definition patterns
    if any(x in q_lower for x in ['what is', 'o que é', 'define', 'definition', 'meaning']):
        return QueryType.DEFINITION
    
    # Requirement patterns
    if any(x in q_lower for x in ['requirement', 'shall', 'must', 'norma', 'standard']):
        return QueryType.REQUIREMENT
    
    return QueryType.GENERIC


# =========================================================
# STAGE 1: STRUCTURED FACT EXTRACTION
# =========================================================

def extract_structured_facts(
    question: str,
    query_type: str,
    chunks: List[Dict],
    ask_ollama_fn,
    max_retries: int = 2
) -> Tuple[Optional[Dict], str]:
    """
    STAGE 1: Extract structured facts as JSON.
    
    Returns:
        (facts_dict, raw_response) or (None, error_msg)
    """
    
    # Build context from chunks
    context = _build_context(chunks)
    
    # Select extraction prompt based on query type
    if query_type == QueryType.ENUMERATION:
        prompt = FACT_EXTRACTION_ENUMERATION.format(question=question, context=context)
        schema = ENUMERATION_SCHEMA
    elif query_type == QueryType.TABLE_DATA:
        prompt = FACT_EXTRACTION_TABLE.format(question=question, context=context)
        schema = TABLE_SCHEMA
    elif query_type == QueryType.DEFINITION:
        prompt = FACT_EXTRACTION_DEFINITION.format(question=question, context=context)
        schema = DEFINITION_SCHEMA
    elif query_type == QueryType.REQUIREMENT:
        prompt = FACT_EXTRACTION_REQUIREMENT.format(question=question, context=context)
        schema = REQUIREMENT_SCHEMA
    else:
        # Generic extraction (no strict schema)
        prompt = f"""Extract key facts from sources to answer: {question}

                    Output as JSON with keys: "facts" (array of strings), "sources" (array), "confidence" (HIGH/MEDIUM/LOW)

                    SOURCES:
                    {context}

                    JSON OUTPUT:"""
        schema = None
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            logger.info(f"Stage 1: Extracting facts (attempt {attempt+1}/{max_retries})")
            
            # Call LLM
            raw_response = ask_ollama_fn(prompt, temperature=0.0, timeout=60)
            
            # Parse JSON
            facts = _parse_json_response(raw_response)
            
            if facts is None:
                logger.warning(f"Failed to parse JSON (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None, f"JSON parsing failed after {max_retries} attempts"
            
            # Validate schema
            if schema:
                try:
                    validate(instance=facts, schema=schema)
                    logger.info(f"✓ Facts extracted and validated: {query_type}")
                except ValidationError as e:
                    logger.warning(f"Schema validation failed (attempt {attempt+1}): {e.message}")
                    if attempt < max_retries - 1:
                        # Add validation error to prompt for retry
                        prompt += f"\n\nPREVIOUS ERROR: {e.message}\nPlease fix the JSON format."
                        continue
                    else:
                        return None, f"Schema validation failed: {e.message}"
            
            return facts, raw_response
            
        except Exception as e:
            logger.error(f"Fact extraction error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, f"Extraction failed: {str(e)}"
    
    return None, "Unknown error in fact extraction"


def _build_context(chunks: List[Dict]) -> str:
    """Build context string from chunks with metadata."""
    # Import here to avoid circular imports
    from utils.table_context import build_context
    return build_context(chunks)


def _parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response (handles markdown fences)."""
    # Remove markdown fences
    cleaned = re.sub(r'```json\s*', '', response)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()
    
    # Try to find JSON block
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"_parse_json_response: JSON decode error: {e}")
        return None


# =========================================================
# STAGE 2: ANSWER GENERATION FROM FACTS
# =========================================================

def generate_answer_from_facts(
    question: str,
    query_type: str,
    facts: Dict,
    ask_ollama_fn
) -> str:
    """
    STAGE 2: Generate natural language answer from structured facts.
    
    CRITICAL: No new information allowed - only reformats facts.
    """
    
    facts_json = json.dumps(facts, indent=2, ensure_ascii=False)
    
    # Select answer prompt based on query type
    if query_type == QueryType.ENUMERATION:
        prompt = ANSWER_FROM_FACTS_ENUMERATION.format(
            question=question,
            facts_json=facts_json
        )
    elif query_type == QueryType.TABLE_DATA:
        prompt = ANSWER_FROM_FACTS_TABLE.format(
            question=question,
            facts_json=facts_json
        )
    else:
        prompt = ANSWER_FROM_FACTS_GENERIC.format(
            question=question,
            facts_json=facts_json
        )
    
    try:
        logger.info(f"Stage 2: Generating answer from facts")
        answer = ask_ollama_fn(prompt, temperature=0.1, timeout=60)
        return answer.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        # Fallback: format facts as bullet points
        return _format_facts_as_fallback(facts, query_type)


def _format_facts_as_fallback(facts: Dict, query_type: str) -> str:
    """Fallback answer formatter if LLM fails."""
    
    if query_type == QueryType.ENUMERATION and "items" in facts:
        lines = ["Based on the retrieved information:\n"]
        for i, item in enumerate(facts["items"], 1):
            lines.append(f"{i}. {item.get('id', 'Item')}: {item.get('description', 'N/A')}")
            lines.append(f"   [Source: {item.get('source', 'Unknown')}]")
        return "\n".join(lines)
    
    elif query_type == QueryType.TABLE_DATA and "table_data" in facts:
        lines = ["Table Data:\n"]
        headers = facts.get("headers", [])
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in facts["table_data"]:
            values = [str(row.get(h, "N/A")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
    
    else:
        # Generic fallback
        return json.dumps(facts, indent=2, ensure_ascii=False)


# =========================================================
# POST-GENERATION VALIDATION
# =========================================================

def validate_answer(
    question: str,
    answer: str,
    facts: Dict,
    query_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate generated answer against extracted facts.
    
    Returns:
        (is_valid, error_message)
    """
    
    if query_type == QueryType.ENUMERATION:
        return _validate_enumeration_answer(question, answer, facts)
    elif query_type == QueryType.TABLE_DATA:
        return _validate_table_answer(answer, facts)
    else:
        # Generic validation: check for hedging phrases
        return _validate_generic_answer(answer)


def _validate_enumeration_answer(question: str, answer: str, facts: Dict) -> Tuple[bool, Optional[str]]:
    """Validate enumeration answer completeness."""
    
    expected_count = facts.get("total_count", 0)
    items = facts.get("items", [])
    
    # Check if all items are mentioned - just warn, don't fail
    # (LLM may paraphrase or combine items)
    missing_items = []
    for item in items:
        item_id = item.get("id", "")
        if item_id and item_id.lower() not in answer.lower():
            missing_items.append(item_id)
    
    if missing_items and len(missing_items) == len(items):
        # Only fail if ALL items are missing (likely wrong extraction)
        return False, f"Missing items: {', '.join(missing_items)}"
    
    # Check count if question asks "how many"
    if re.search(r'\bhow many\b|\bquantos\b', question, re.IGNORECASE):
        # Extract number from answer
        numbers = re.findall(r'\b(\d+)\b', answer)
        if numbers:
            stated_count = int(numbers[0])
            if stated_count != expected_count:
                return False, f"Count mismatch: stated {stated_count}, expected {expected_count}"
    
    return True, None


def _validate_table_answer(answer: str, facts: Dict) -> Tuple[bool, Optional[str]]:
    """Validate table completeness."""
    
    expected_rows = facts.get("row_count", 0)
    
    # Count rows in answer (heuristic: count pipe symbols or numbered items)
    answer_rows = answer.count("|") - answer.count("---")  # Markdown tables
    if answer_rows < expected_rows:
        return False, f"Incomplete table: {answer_rows} rows in answer, {expected_rows} expected"
    
    return True, None


def _validate_generic_answer(answer: str) -> Tuple[bool, Optional[str]]:
    """Validate generic answer for hedging phrases."""
    
    hedging_phrases = [
        "i was unable to find",
        "i couldn't find",
        "no information",
        "however, i found",
        "but i found",
    ]
    
    answer_lower = answer.lower()
    for phrase in hedging_phrases:
        if phrase in answer_lower:
            return False, f"Answer contains hedging phrase: '{phrase}'"
    
    return True, None


# =========================================================
# MAIN TWO-STAGE PIPELINE
# =========================================================

def generate_structured_answer(
    question: str,
    chunks: List[Dict],
    ask_ollama_fn,
    max_retries: int = 2
) -> Dict:
    """
    Main two-stage answer generation pipeline with complementary responses.
    """
    
    # STEP 0: Detect query type
    query_type = detect_query_type(question, ask_ollama_fn)
    if DEBUGS:
        print(f"\ngenerate_structured_answer: query_type: {query_type}")

    # STEP 1: Extract structured facts
    facts, raw_facts = extract_structured_facts(
        question, query_type, chunks, ask_ollama_fn, max_retries
    )
    if DEBUGS:
        print(f"\ngenerate_structured_answer: facts: {facts}\n\n raw_facts: {raw_facts}")

    if facts is None:
        logger.error(f"Fact extraction failed: {raw_facts}")
        return {
            "answer": f"Extraction failed: {raw_facts}",
            "facts": {},
            "query_type": query_type,
            "confidence": "LOW",
            "validation_passed": False,
            "error": raw_facts
        }
    
    # STEP 2: Generate answer from facts
    answer = generate_answer_from_facts(question, query_type, facts, ask_ollama_fn)
    
    # STEP 2.5: Generate complementary response for richer context
    if query_type in [QueryType.TABLE_DATA, QueryType.ENUMERATION, QueryType.REQUIREMENT]:
        complement = _generate_complement(question, chunks, ask_ollama_fn)
        if complement:
            answer = f"{answer}\n\n**Additional Context:**\n{complement}"
    
    if DEBUGS:
        print(f"\ngenerate_structured_answer: answer: {answer}")

    # STEP 3: Validate answer (non-blocking)
    is_valid, error_msg = validate_answer(question, answer, facts, query_type)
    if DEBUGS:
        print(f"\ngenerate_structured_answer: is_valid: {is_valid}\n\n error_msg: {error_msg}")
    
    if not is_valid:
        logger.warning(f"Validation warning: {error_msg}")

    confidence = facts.get("extraction_confidence", "MEDIUM")
    
    return {
        "answer": answer,
        "facts": facts,
        "query_type": query_type,
        "confidence": confidence,
        "validation_passed": is_valid,
        "validation_error": error_msg if not is_valid else None
    }


def _generate_complement(question: str, chunks: List[Dict], ask_ollama_fn) -> Optional[str]:
    """Generate complementary context (notes, exceptions, standards not in tables)."""
    
    context = _build_context(chunks)
    
    prompt = f"""Extract ONLY additional context NOT found in tables/lists for: {question}

Focus on:
- Notes, exceptions, warnings in the text
- Referenced standards (IEC, BS EN, etc.)
- Conditions or constraints
- Brief explanations of terms

IMPORTANT: Cite each piece of info with [Source: filename - Page X]

SOURCES:
{context}

Keep response under 100 words. If no additional context, respond "None".

CONTEXT:"""

    try:
        response = ask_ollama_fn(prompt, temperature=0.1, timeout=30)
        response = response.strip()
        
        if not response or response.lower() in ["none", "none.", "n/a"] or len(response) < 20:
            return None
            
        return response
    except Exception as e:
        logger.warning(f"Complement generation failed: {e}")
        return None


# =========================================================
# HELPER: ENHANCED ASK_OLLAMA WITH TIMEOUT
# =========================================================

def ask_ollama_safe(
    prompt: str,
    model: str = "llama3.1:8b",
    temperature: float = 0.15,
    timeout: int = 60,
    ollama_url: str = "http://localhost:11434/api/generate"
) -> str:
    """
    Safe Ollama call with timeout protection.
    
    FIXES: Added explicit timeout to prevent indefinite hangs.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.1,
            "top_k": 10,
        }
    }
    
    try:
        resp = requests.post(ollama_url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.Timeout:
        logger.error(f"Ollama request timed out after {timeout}s")
        raise TimeoutError(f"LLM request exceeded {timeout}s timeout")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise