from abc import ABC, abstractmethod
from .schemas import ToolRequest, ToolResult


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, request: ToolRequest) -> ToolResult:
        pass