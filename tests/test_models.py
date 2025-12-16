"""Tests for data models."""

from datetime import datetime
from backend.app.models.resume import Resume, ResumeSection, SectionType
from backend.app.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    AgentType,
)


class TestResumeModels:
    """Tests for resume-related models."""

    def test_resume_section_creation(self):
        """Test creating a resume section."""
        section = ResumeSection(
            section_type=SectionType.EXPERIENCE,
            title="Work Experience",
            content="Software Engineer at Company X",
            order=1,
        )

        assert section.section_type == SectionType.EXPERIENCE
        assert section.title == "Work Experience"
        assert section.order == 1

    def test_resume_creation(self):
        """Test creating a resume."""
        resume = Resume(
            id="test-123",
            user_id="user-456",
            filename="resume.pdf",
            raw_text="Test resume content",
            sections=[],
        )

        assert resume.id == "test-123"
        assert resume.user_id == "user-456"
        assert resume.filename == "resume.pdf"
        assert isinstance(resume.created_at, datetime)

    def test_resume_get_section(self):
        """Test getting a specific section from resume."""
        sections = [
            ResumeSection(
                section_type=SectionType.SUMMARY,
                title="Summary",
                content="Professional summary",
                order=0,
            ),
            ResumeSection(
                section_type=SectionType.EXPERIENCE,
                title="Experience",
                content="Work history",
                order=1,
            ),
        ]

        resume = Resume(
            id="test-123",
            user_id="user-456",
            filename="resume.pdf",
            raw_text="Test",
            sections=sections,
        )

        summary = resume.get_section(SectionType.SUMMARY)
        assert summary is not None
        assert summary.title == "Summary"

        skills = resume.get_section(SectionType.SKILLS)
        assert skills is None

    def test_resume_get_full_text(self):
        """Test getting full resume text."""
        sections = [
            ResumeSection(
                section_type=SectionType.SUMMARY,
                title="Summary",
                content="Professional summary",
                order=0,
            ),
            ResumeSection(
                section_type=SectionType.EXPERIENCE,
                title="Experience",
                content="Work history",
                order=1,
            ),
        ]

        resume = Resume(
            id="test-123",
            user_id="user-456",
            filename="resume.pdf",
            raw_text="Raw text",
            sections=sections,
        )

        full_text = resume.get_full_text()
        assert "Summary" in full_text
        assert "Experience" in full_text


class TestConversationModels:
    """Tests for conversation-related models."""

    def test_message_creation(self):
        """Test creating a message."""
        message = Message(id="msg-123", role=MessageRole.USER, content="Hello")

        assert message.id == "msg-123"
        assert message.role == MessageRole.USER
        assert message.content == "Hello"

    def test_conversation_creation(self):
        """Test creating a conversation."""
        conversation = Conversation(id="conv-123", user_id="user-456")

        assert conversation.id == "conv-123"
        assert conversation.user_id == "user-456"
        assert len(conversation.messages) == 0

    def test_conversation_add_message(self):
        """Test adding a message to conversation."""
        conversation = Conversation(id="conv-123", user_id="user-456")

        message = Message(id="msg-123", role=MessageRole.USER, content="Hello")

        conversation.add_message(message)

        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "Hello"

    def test_conversation_get_history(self):
        """Test getting conversation history with limit."""
        conversation = Conversation(id="conv-123", user_id="user-456")

        for i in range(5):
            message = Message(
                id=f"msg-{i}", role=MessageRole.USER, content=f"Message {i}"
            )
            conversation.add_message(message)

        # Get all messages
        all_messages = conversation.get_history()
        assert len(all_messages) == 5

        # Get limited messages
        limited = conversation.get_history(limit=3)
        assert len(limited) == 3
        assert limited[0].content == "Message 2"  # Last 3 messages

    def test_agent_types(self):
        """Test agent type enumeration."""
        assert AgentType.COMPANY_RESEARCH.value == "company_research"
        assert AgentType.JOB_MATCHING.value == "job_matching"
        assert AgentType.TRANSLATION.value == "translation"
