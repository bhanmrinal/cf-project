"""Conversation and message data models."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentType(str, Enum):
    """Types of specialized agents."""

    ROUTER = "router"
    COMPANY_RESEARCH = "company_research"
    JOB_MATCHING = "job_matching"
    TRANSLATION = "translation"
    SECTION_ENHANCEMENT = "section_enhancement"


def _utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class Message(BaseModel):
    """A single message in a conversation."""

    id: str
    role: MessageRole
    content: str
    agent_type: AgentType | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    # For tracking agent reasoning and actions
    reasoning: str | None = None
    actions_taken: list[dict] = Field(default_factory=list)


class Conversation(BaseModel):
    """A conversation session with a user."""

    id: str
    user_id: str
    resume_id: str | None = None
    messages: list[Message] = Field(default_factory=list)
    current_resume_version: int = 1
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def get_history(self, limit: int | None = None) -> list[Message]:
        """Get conversation history, optionally limited to recent messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        context_parts = []
        if self.resume_id:
            context_parts.append(f"Resume ID: {self.resume_id}")
        if self.context.get("target_company"):
            context_parts.append(f"Target Company: {self.context['target_company']}")
        if self.context.get("target_role"):
            context_parts.append(f"Target Role: {self.context['target_role']}")
        if self.context.get("target_language"):
            context_parts.append(f"Target Language: {self.context['target_language']}")
        return " | ".join(context_parts) if context_parts else "No context set"
