"""
Base Agent Module.

Provides the abstract base class for all specialized agents with common
functionality for LLM interaction, context management, and result formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from backend.app.core.llm import get_llm
from backend.app.models.resume import Resume, ResumeSection
from backend.app.models.conversation import Conversation, AgentType


@dataclass
class AgentResult:
    """Result returned by an agent after processing."""

    success: bool
    message: str
    updated_resume: Optional[Resume] = None
    updated_sections: list[ResumeSection] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.

    Provides common functionality for:
    - LLM initialization and interaction
    - Context management
    - Prompt formatting
    - Result standardization
    """

    agent_type: AgentType = AgentType.ROUTER
    description: str = "Base agent"

    def __init__(self, temperature: float = 0.7):
        """
        Initialize the agent.

        Args:
            temperature: LLM sampling temperature.
        """
        self._llm: Optional[BaseChatModel] = None
        self.temperature = temperature

    @property
    def llm(self) -> BaseChatModel:
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = get_llm(self.temperature)
        return self._llm

    @abstractmethod
    async def process(
        self,
        user_message: str,
        resume: Resume,
        conversation: Conversation,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Process a user request and return the result.

        Args:
            user_message: The user's message/request.
            resume: The current resume being optimized.
            conversation: The conversation context.
            context: Additional context (e.g., target company, job description).

        Returns:
            AgentResult with the processing outcome.
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    async def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """
        Invoke the LLM with the given prompts.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message/query.
            conversation_history: Optional list of (role, content) tuples.

        Returns:
            LLM response content.
        """
        messages = [SystemMessage(content=system_prompt)]

        if conversation_history:
            for role, content in conversation_history:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_prompt))

        response = await self.llm.ainvoke(messages)
        return response.content

    def _format_resume_for_prompt(self, resume: Resume) -> str:
        """
        Format resume content for inclusion in prompts.

        Args:
            resume: The resume to format.

        Returns:
            Formatted resume string.
        """
        if resume.sections:
            sections_text = []
            for section in sorted(resume.sections, key=lambda s: s.order):
                sections_text.append(f"## {section.title}\n{section.content}")
            return "\n\n".join(sections_text)
        return resume.raw_text

    def _extract_sections_from_response(
        self, response: str, original_resume: Resume
    ) -> list[ResumeSection]:
        """
        Extract resume sections from LLM response.

        Args:
            response: LLM response containing updated resume.
            original_resume: Original resume for reference.

        Returns:
            List of updated ResumeSection objects.
        """
        import re
        from backend.app.models.resume import SectionType

        sections = []
        section_pattern = r"##\s*(.+?)\n([\s\S]*?)(?=##\s*|\Z)"
        matches = re.findall(section_pattern, response)

        type_mapping = {
            "contact": SectionType.CONTACT,
            "summary": SectionType.SUMMARY,
            "objective": SectionType.SUMMARY,
            "profile": SectionType.SUMMARY,
            "experience": SectionType.EXPERIENCE,
            "work": SectionType.EXPERIENCE,
            "education": SectionType.EDUCATION,
            "skills": SectionType.SKILLS,
            "projects": SectionType.PROJECTS,
            "certifications": SectionType.CERTIFICATIONS,
            "languages": SectionType.LANGUAGES,
        }

        for i, (title, content) in enumerate(matches):
            title = title.strip()
            content = content.strip()

            section_type = SectionType.OTHER
            title_lower = title.lower()
            for key, stype in type_mapping.items():
                if key in title_lower:
                    section_type = stype
                    break

            sections.append(
                ResumeSection(
                    section_type=section_type, title=title, content=content, order=i
                )
            )

        if not sections and original_resume.sections:
            return original_resume.sections

        return sections

    def _identify_changes(
        self,
        original_sections: list[ResumeSection],
        updated_sections: list[ResumeSection],
    ) -> list[dict[str, Any]]:
        """
        Identify changes between original and updated sections.

        Args:
            original_sections: Original resume sections.
            updated_sections: Updated resume sections.

        Returns:
            List of change dictionaries.
        """
        changes = []
        original_by_type = {s.section_type: s for s in original_sections}
        updated_by_type = {s.section_type: s for s in updated_sections}

        for section_type, updated in updated_by_type.items():
            original = original_by_type.get(section_type)

            if original is None:
                changes.append(
                    {
                        "section": updated.title,
                        "type": "added",
                        "new_content": updated.content,
                    }
                )
            elif original.content != updated.content:
                changes.append(
                    {
                        "section": updated.title,
                        "type": "modified",
                        "original_content": original.content[:200] + "..."
                        if len(original.content) > 200
                        else original.content,
                        "new_content": updated.content[:200] + "..."
                        if len(updated.content) > 200
                        else updated.content,
                    }
                )

        for section_type, original in original_by_type.items():
            if section_type not in updated_by_type:
                changes.append(
                    {
                        "section": original.title,
                        "type": "removed",
                        "original_content": original.content[:200] + "..."
                        if len(original.content) > 200
                        else original.content,
                    }
                )

        return changes
