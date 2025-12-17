"""
Firebase Service.

Handles all Firebase Firestore operations for storing conversations,
resumes, and resume versions.
"""

from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import get_settings
from app.models.conversation import (
    AgentType,
    Conversation,
    Message,
    MessageRole,
)
from app.models.resume import Resume, ResumeSection, ResumeVersion


class FirebaseService:
    """Service for Firebase Firestore operations."""

    def __init__(self):
        self.settings = get_settings()
        self._db = None
        self._initialized = False

    def _get_db(self):
        """Lazy initialization of Firebase client."""
        if self._db is not None:
            return self._db

        credentials = self.settings.firebase_credentials
        if not credentials:
            return None

        try:
            import firebase_admin
            from firebase_admin import credentials as fb_credentials
            from firebase_admin import firestore

            if not firebase_admin._apps:
                cred = fb_credentials.Certificate(credentials)
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            self._initialized = True
            return self._db
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if Firebase is available."""
        return self._get_db() is not None

    # ==================== Conversation Operations ====================

    async def create_conversation(
        self, user_id: str, resume_id: str | None = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            id=str(uuid4()),
            user_id=user_id,
            resume_id=resume_id,
            messages=[],
            context={},
        )

        db = self._get_db()
        if db:
            doc_ref = db.collection("conversations").document(conversation.id)
            doc_ref.set(self._conversation_to_dict(conversation))

        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        db = self._get_db()
        if not db:
            return None

        doc_ref = db.collection("conversations").document(conversation_id)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        return self._dict_to_conversation(doc.to_dict())

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation."""
        conversation.updated_at = datetime.now(UTC)

        db = self._get_db()
        if db:
            doc_ref = db.collection("conversations").document(conversation.id)
            doc_ref.update(self._conversation_to_dict(conversation))

        return conversation

    async def add_message_to_conversation(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        agent_type: AgentType | None = None,
        reasoning: str | None = None,
        actions: list[dict] | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        message = Message(
            id=str(uuid4()),
            role=role,
            content=content,
            agent_type=agent_type,
            reasoning=reasoning,
            actions_taken=actions or [],
        )

        db = self._get_db()
        if db:
            from firebase_admin import firestore

            doc_ref = db.collection("conversations").document(conversation_id)
            doc_ref.update(
                {
                    "messages": firestore.ArrayUnion([self._message_to_dict(message)]),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )

        return message

    async def get_user_conversations(self, user_id: str) -> list[Conversation]:
        """Get all conversations for a user."""
        db = self._get_db()
        if not db:
            return []

        try:
            from google.cloud.firestore_v1.base_query import FieldFilter

            docs = (
                db.collection("conversations")
                .where(filter=FieldFilter("user_id", "==", user_id))
                .stream()
            )
        except ImportError:
            # Fallback for older versions
            docs = (
                db.collection("conversations").where("user_id", "==", user_id).stream()
            )

        return [self._dict_to_conversation(doc.to_dict()) for doc in docs]

    # ==================== Resume Operations ====================

    async def save_resume(self, resume: Resume) -> Resume:
        """Save a resume to Firestore."""
        db = self._get_db()
        if db:
            doc_ref = db.collection("resumes").document(resume.id)
            doc_ref.set(self._resume_to_dict(resume))

        return resume

    async def get_resume(self, resume_id: str) -> Resume | None:
        """Get a resume by ID."""
        db = self._get_db()
        if not db:
            return None

        doc_ref = db.collection("resumes").document(resume_id)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        return self._dict_to_resume(doc.to_dict())

    async def update_resume(self, resume: Resume) -> Resume:
        """Update an existing resume."""
        resume.updated_at = datetime.now(UTC)

        db = self._get_db()
        if db:
            doc_ref = db.collection("resumes").document(resume.id)
            doc_ref.update(self._resume_to_dict(resume))

        return resume

    # ==================== Resume Version Operations ====================

    async def create_resume_version(
        self,
        resume_id: str,
        content: str,
        sections: list[ResumeSection],
        changes_description: str,
        agent_used: str | None = None,
        parent_version_id: str | None = None,
    ) -> ResumeVersion:
        """Create a new resume version."""
        version_number = await self._get_next_version_number(resume_id)

        version = ResumeVersion(
            id=str(uuid4()),
            resume_id=resume_id,
            version_number=version_number,
            content=content,
            sections=sections,
            changes_description=changes_description,
            agent_used=agent_used,
            parent_version_id=parent_version_id,
        )

        db = self._get_db()
        if db:
            doc_ref = db.collection("resume_versions").document(version.id)
            doc_ref.set(self._version_to_dict(version))

        return version

    async def get_resume_versions(self, resume_id: str) -> list[ResumeVersion]:
        """Get all versions of a resume, sorted by version number."""
        db = self._get_db()
        if not db:
            return []

        try:
            from google.cloud.firestore_v1.base_query import FieldFilter

            # Query without order_by to avoid needing composite index
            # We'll sort in Python instead
            docs = (
                db.collection("resume_versions")
                .where(filter=FieldFilter("resume_id", "==", resume_id))
                .stream()
            )
            versions = [self._dict_to_version(doc.to_dict()) for doc in docs]
        except ImportError:
            # Fallback for older versions
            try:
                docs = (
                    db.collection("resume_versions")
                    .where("resume_id", "==", resume_id)
                    .stream()
                )
                versions = [self._dict_to_version(doc.to_dict()) for doc in docs]
            except Exception as e:
                print(f"Error fetching resume versions: {e}")
                return []
        except Exception as e:
            print(f"Error fetching resume versions: {e}")
            return []

        # Sort in Python to avoid needing composite index
        return sorted(versions, key=lambda v: v.version_number)

    async def get_resume_version(self, version_id: str) -> ResumeVersion | None:
        """Get a specific resume version."""
        db = self._get_db()
        if not db:
            return None

        doc_ref = db.collection("resume_versions").document(version_id)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        return self._dict_to_version(doc.to_dict())

    async def _get_next_version_number(self, resume_id: str) -> int:
        """Get the next version number for a resume."""
        versions = await self.get_resume_versions(resume_id)
        if not versions:
            return 1
        return max(v.version_number for v in versions) + 1

    # ==================== Serialization Helpers ====================

    def _conversation_to_dict(self, conv: Conversation) -> dict:
        """Convert Conversation to dictionary."""
        return {
            "id": conv.id,
            "user_id": conv.user_id,
            "resume_id": conv.resume_id,
            "messages": [self._message_to_dict(m) for m in conv.messages],
            "current_resume_version": conv.current_resume_version,
            "context": conv.context,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        }

    def _dict_to_conversation(self, data: dict) -> Conversation:
        """Convert dictionary to Conversation."""
        return Conversation(
            id=data["id"],
            user_id=data["user_id"],
            resume_id=data.get("resume_id"),
            messages=[self._dict_to_message(m) for m in data.get("messages", [])],
            current_resume_version=data.get("current_resume_version", 1),
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _message_to_dict(self, msg: Message) -> dict:
        """Convert Message to dictionary."""
        return {
            "id": msg.id,
            "role": msg.role.value,
            "content": msg.content,
            "agent_type": msg.agent_type.value if msg.agent_type else None,
            "metadata": msg.metadata,
            "created_at": msg.created_at.isoformat(),
            "reasoning": msg.reasoning,
            "actions_taken": msg.actions_taken,
        }

    def _dict_to_message(self, data: dict) -> Message:
        """Convert dictionary to Message."""
        return Message(
            id=data["id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            agent_type=(
                AgentType(data["agent_type"]) if data.get("agent_type") else None
            ),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            reasoning=data.get("reasoning"),
            actions_taken=data.get("actions_taken", []),
        )

    def _resume_to_dict(self, resume: Resume) -> dict:
        """Convert Resume to dictionary."""
        return {
            "id": resume.id,
            "user_id": resume.user_id,
            "filename": resume.filename,
            "raw_text": resume.raw_text,
            "sections": [self._section_to_dict(s) for s in resume.sections],
            "metadata": resume.metadata,
            "created_at": resume.created_at.isoformat(),
            "updated_at": resume.updated_at.isoformat(),
        }

    def _dict_to_resume(self, data: dict) -> Resume:
        """Convert dictionary to Resume."""
        return Resume(
            id=data["id"],
            user_id=data["user_id"],
            filename=data["filename"],
            raw_text=data["raw_text"],
            sections=[self._dict_to_section(s) for s in data.get("sections", [])],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _section_to_dict(self, section: ResumeSection) -> dict:
        """Convert ResumeSection to dictionary."""
        return {
            "section_type": section.section_type.value,
            "title": section.title,
            "content": section.content,
            "order": section.order,
            "metadata": section.metadata,
        }

    def _dict_to_section(self, data: dict) -> ResumeSection:
        """Convert dictionary to ResumeSection."""
        from app.models.resume import SectionType

        return ResumeSection(
            section_type=SectionType(data["section_type"]),
            title=data["title"],
            content=data["content"],
            order=data.get("order", 0),
            metadata=data.get("metadata", {}),
        )

    def _version_to_dict(self, version: ResumeVersion) -> dict:
        """Convert ResumeVersion to dictionary."""
        return {
            "id": version.id,
            "resume_id": version.resume_id,
            "version_number": version.version_number,
            "content": version.content,
            "sections": [self._section_to_dict(s) for s in version.sections],
            "changes_description": version.changes_description,
            "agent_used": version.agent_used,
            "created_at": version.created_at.isoformat(),
            "parent_version_id": version.parent_version_id,
        }

    def _dict_to_version(self, data: dict) -> ResumeVersion:
        """Convert dictionary to ResumeVersion."""
        return ResumeVersion(
            id=data["id"],
            resume_id=data["resume_id"],
            version_number=data["version_number"],
            content=data["content"],
            sections=[self._dict_to_section(s) for s in data.get("sections", [])],
            changes_description=data.get("changes_description", ""),
            agent_used=data.get("agent_used"),
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_version_id=data.get("parent_version_id"),
        )


class InMemoryStore:
    """In-memory storage for development/testing when Firebase is unavailable."""

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self.resumes: dict[str, Resume] = {}
        self.versions: dict[str, ResumeVersion] = {}

    async def create_conversation(
        self, user_id: str, resume_id: str | None = None
    ) -> Conversation:
        conversation = Conversation(
            id=str(uuid4()),
            user_id=user_id,
            resume_id=resume_id,
        )
        self.conversations[conversation.id] = conversation
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        conversation.updated_at = datetime.now(UTC)
        self.conversations[conversation.id] = conversation
        return conversation

    async def save_resume(self, resume: Resume) -> Resume:
        self.resumes[resume.id] = resume
        return resume

    async def get_resume(self, resume_id: str) -> Resume | None:
        return self.resumes.get(resume_id)

    async def update_resume(self, resume: Resume) -> Resume:
        """Update an existing resume."""
        resume.updated_at = datetime.now(UTC)
        self.resumes[resume.id] = resume
        return resume

    async def create_resume_version(
        self,
        resume_id: str,
        content: str,
        sections: list[ResumeSection],
        changes_description: str,
        agent_used: str | None = None,
        parent_version_id: str | None = None,
    ) -> ResumeVersion:
        existing_versions = [
            v for v in self.versions.values() if v.resume_id == resume_id
        ]
        version_number = (
            max((v.version_number for v in existing_versions), default=0) + 1
        )

        version = ResumeVersion(
            id=str(uuid4()),
            resume_id=resume_id,
            version_number=version_number,
            content=content,
            sections=sections,
            changes_description=changes_description,
            agent_used=agent_used,
            parent_version_id=parent_version_id,
        )
        self.versions[version.id] = version
        return version

    async def get_resume_versions(self, resume_id: str) -> list[ResumeVersion]:
        return sorted(
            [v for v in self.versions.values() if v.resume_id == resume_id],
            key=lambda v: v.version_number,
        )


_storage_instance = None


def get_storage_service():
    """Get the appropriate storage service based on configuration (singleton)."""
    global _storage_instance
    if _storage_instance is None:
        firebase_service = FirebaseService()
        if firebase_service.is_available:
            _storage_instance = firebase_service
        else:
            _storage_instance = InMemoryStore()
    return _storage_instance
