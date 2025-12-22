"""
Centralized prompt templates for the RAG system.
"""

INTENT_CLASSIFICATION_PROMPT = """You are classifying user queries for a technical documentation RAG system.

CONVERSATION HISTORY:
{recent_context}

CURRENT QUERY: "{query}"

INTENT TYPES WITH EXAMPLES:

1. FACTUAL - Asking for definitions, specifications, or facts
   Examples: "What is X?", "Define Type 1 terminals", "What does <technical_document> specify?"

2. PROCEDURAL - Asking how to perform an action or follow steps
   Examples: "How do I install?", "What are the steps to?", "How to configure?"

3. COMPARATIVE - Comparing multiple options or alternatives
   Examples: "<some_information> vs <other_information>?", "Difference between X and Y?"

4. CLARIFICATION - Requesting more detail about the previous answer
   Examples: "What do you mean?", "Explain that more", "Can you elaborate?"

5. EXPLORATORY - Open-ended topic exploration
   Examples: "Tell me about <some_topic>", "What should I know about?", "Overview of X"

6. FOLLOWUP - Following up on previous conversation topic
   Examples: "What about <info>?" (after discussing <other_info>), "And the other one?"

7. LIST - Requesting enumerated items or categories
   Examples: "List all <info>", "What options are there?", "Show me the categories"

RESPOND IN EXACTLY THIS FORMAT:
reasoning: <your step-by-step analysis>
intent: <one of: factual, procedural, comparative, clarification, exploratory, followup, list>
confidence: <number between 0.0 and 1.0>"""


HYDE_PROMPT = """You are a technical documentation expert for electrical engineering specifications.
Given this question, write a short paragraph (2-3 sentences) that might appear in a technical specification document answering this question.

Include:
- Specific technical terms (e.g., terminal blocks, enclosures, cable glands)
- Type classifications (Type 1, Type 2, Class A, Category B, etc.)
- Standards references (BS EN, IEC, IEEE)
- Specification language (shall be used, is required, must comply)

Question: {query}

Technical specification excerpt:"""


KEYWORD_EXTRACTION_PROMPT = """Given this question about electrical/engineering specifications, list the most likely technical terms that would appear in specification documents.

Include variations like:
- Component names (terminal blocks, cable glands, enclosures, splice boxes)
- Type classifications (Type 1, Type 2, Type "1", Class A)
- Industry terms (screw clamp, spring retention, DIN rail)
- Related equipment and standards
- Organization names 

Question: {query}

Return only keywords/phrases, one per line, no explanations or numbering:"""


AMBIGUITY_DETECTION_PROMPT = """Analyze if this query is too ambiguous to search technical documentation effectively.

Previous Conversation:
{recent_context}

Previously Mentioned Topics: {entities}

Current Query: "{query}"

A query is ambiguous if:
1. It uses vague words without context (e.g., "that thing", "the other one")
2. It could refer to multiple completely different topics
3. Critical information is missing to understand what's being asked
4. It uses pronouns without clear reference AND there's no conversation history

A query is NOT ambiguous if:
1. It's a followup and the topic is clear from conversation history
2. It uses general terms but the domain is clear (e.g., "terminals" in electrical context)
3. It's exploratory but the subject area is defined

Respond in EXACTLY this format:
ambiguous: yes/no
clarification: <question to ask user if ambiguous, otherwise "none">"""


QUERY_EXPANSION_PROMPT = """Rewrite this query to be more specific and searchable for technical documentation.

Conversation Context:
{recent_context}

Previously Mentioned Topics: {entities}

Current Query: "{query}"
Query Intent: {intent}

Instructions:
- If this is a followup question, incorporate what was discussed before
- If terms are ambiguous, expand based on conversation context
- Add relevant technical synonyms
- Keep the query focused and natural
- If already specific, just return it slightly improved

Expanded Query (single line):"""


INTERATIVE_REFINEMENT_PROMPT = """Based on these document excerpts, suggest a refined search query that might find more relevant information.

Original Query: {original_query}
Current Query: {current_query}

Found Document Excerpts:
{found_texts}

Generate a single refined search query, the most concise it can be without losing meaning:"""


GET_ANSWER_GENERATION_PROMPT = """You are a helpful technical assistant answering questions about engineering specifications and documentation.

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: {question}

INSTRUCTIONS:
{instruction}

IMPORTANT:
- Base your answer ONLY on the provided documentation context
- Always cite sources using [Source X] format
- If information is incomplete or not found, acknowledge it honestly
- If this is a followup question, connect to previous discussion naturally
- Be conversational but accurate
- Use bullet points or numbered lists for clarity when appropriate

DOCUMENTATION CONTEXT:
{context}

ANSWER:"""