"""
CONVERSATION MEMORY

These classes track conversation history so the system can handle followup
questions like "what about Type 2?" after asking about "Type 1 terminals".
This is what makes the system feel like 'ChatGPT'.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

MAX_MESSAGES = 10 # Max memory of messages

@dataclass
class Message:
    """
    Represents a single message in the conversation (either user or assistant).
    
    Stores:
    - What was said
    - Who said it (user or AI)
    - When it was said
    - What documents were cited
    - Metadata (keywords, intent, etc.)
    """
    role: str  # 'user' or 'assistant'
    content: str  # The actual message text
    timestamp: datetime = field(default_factory=datetime.now)  # When this was said
    sources: Optional[List[Dict]] = None  # Documents cited in the answer
    metadata: Optional[Dict] = None  # Extra info (keywords, intent type, etc.)

class ConversationHistory:
    """
    Manages the entire conversation memory - tracks all messages and extracts context.
    
    This allows the system to:
    1. Remember what was discussed earlier
    2. Track technical terms mentioned ("terminal blocks", "Type 1", etc.)
    3. Resolve ambiguous followup questions ("what about the other type?")
    4. Build context for query expansion
    
    Example conversation:
    User: "What are Type 1 terminals?"
    [System remembers: topic=terminals, entity=Type 1]
    User: "And Type 2?" 
    [System knows to search for Type 2 terminals]
    """
    
    def __init__(self) -> None:
        self.messages: List[Message] = []  # All messages in conversation
        self.max_messages = MAX_MESSAGES  # Limit memory to prevent context overflow
        self.extracted_topics: List[str] = []  # Main topics discussed
        self.extracted_entities: List[str] = []  # Technical terms mentioned
    
    def add_message(self, role: str, content: str, sources=None, metadata=None):
        """
        Add a new message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: The message text
            sources: List of documents cited (for assistant messages)
            metadata: Extra info like keywords, intent type, etc.
        """
        msg = Message(
            role=role,
            content=content,
            sources=sources,
            metadata=metadata
        )
        self.messages.append(msg)
        
        # Keep only recent messages to prevent context from getting too long
        # LLMs have token limits, so we can't send entire conversation history
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]  # Keep last N messages
    
    def add_entities(self, entities: List[str]):
        """
        Track technical terms/entities mentioned in conversation.
        
        Example: If user asks about "Type 1 terminal blocks" and "cable glands",
        we store those terms to help with followup questions.
        
        Args:
            entities: List of technical terms to remember
        """
        for entity in entities:
            if entity not in self.extracted_entities:
                self.extracted_entities.append(entity)
        # Keep last 20 entities only (prevent list from growing forever)
        self.extracted_entities = self.extracted_entities[-20:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """
        Get the last N messages formatted as a string for LLM context.
        
        This is sent to the LLM when classifying intent or expanding queries,
        so it can understand what's been discussed.
        
        Args:
            n: Number of recent messages to include
            
        Returns:
            Formatted string like:
            "User: What are Type 1 terminals?
             Assistant: Type 1 terminals are..."
        """
        recent = self.messages[-n:] if self.messages else []
        context_parts = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages to first 300 chars to save tokens
            context_parts.append(f"{role}: {msg.content[:300]}")
        return "\n".join(context_parts)
    
    def get_entities_string(self) -> str:
        """Get mentioned entities as string"""
        return ", ".join(self.extracted_entities[-10:]) if self.extracted_entities else "None"
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.extracted_topics = []
        self.extracted_entities = []