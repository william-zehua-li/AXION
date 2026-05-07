from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolRequest:
    tool_name: str
    query: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    content: Any
    confidence: float = 1.0
    source: Optional[str] = None
    error: Optional[str] = None


@dataclass
class KnowledgeCandidate:
    key: str
    content: str
    confidence: float
    source: Optional[str] = None