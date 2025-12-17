"""Resume data models."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class SectionType(str, Enum):
    """Types of resume sections."""

    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"
    OTHER = "other"


def _utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class ResumeSection(BaseModel):
    """A section within a resume."""

    section_type: SectionType
    title: str
    content: str
    order: int = 0
    metadata: dict = Field(default_factory=dict)


class Resume(BaseModel):
    """Resume data model."""

    id: str
    user_id: str
    filename: str
    raw_text: str
    sections: list[ResumeSection] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def get_section(self, section_type: SectionType) -> ResumeSection | None:
        """Get a specific section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def get_full_text(self) -> str:
        """Get the full resume text from all sections."""
        if self.sections:
            return "\n\n".join(
                f"{section.title}\n{section.content}"
                for section in sorted(self.sections, key=lambda s: s.order)
            )
        return self.raw_text


class ResumeVersion(BaseModel):
    """A version of a resume for tracking changes."""

    id: str
    resume_id: str
    version_number: int
    content: str
    sections: list[ResumeSection] = Field(default_factory=list)
    changes_description: str = ""
    agent_used: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    parent_version_id: str | None = None
