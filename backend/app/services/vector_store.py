"""
Vector Store Service.

Handles ChromaDB operations for semantic search and resume content retrieval.
"""

from uuid import uuid4

import chromadb
from app.core.config import get_settings
from app.core.llm import get_embeddings
from app.models.resume import Resume
from chromadb.config import Settings as ChromaSettings


class VectorStoreService:
    """Service for vector store operations using ChromaDB."""

    def __init__(self):
        self.settings = get_settings()
        self._client: chromadb.Client | None = None
        self._embeddings = None

    @property
    def client(self) -> chromadb.Client:
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            # Use the new PersistentClient API (ChromaDB 0.4+)
            self._client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def embeddings(self):
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    async def index_resume(self, resume: Resume) -> None:
        """
        Index a resume's content for semantic search.

        Args:
            resume: The resume to index.
        """
        collection = self._get_or_create_collection("resumes")

        documents = []
        metadatas = []
        ids = []

        documents.append(resume.raw_text)
        metadatas.append(
            {
                "resume_id": resume.id,
                "user_id": resume.user_id,
                "type": "full_resume",
                "filename": resume.filename,
            }
        )
        ids.append(f"{resume.id}_full")

        for section in resume.sections:
            if section.content.strip():
                documents.append(section.content)
                metadatas.append(
                    {
                        "resume_id": resume.id,
                        "user_id": resume.user_id,
                        "type": "section",
                        "section_type": section.section_type.value,
                        "section_title": section.title,
                    }
                )
                ids.append(
                    f"{resume.id}_{section.section_type.value}_{uuid4().hex[:8]}"
                )

        if documents:
            embeddings = self.embeddings.embed_documents(documents)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

    async def search_resume_content(
        self, query: str, resume_id: str | None = None, n_results: int = 5
    ) -> list[dict]:
        """
        Search for relevant resume content.

        Args:
            query: Search query.
            resume_id: Optional resume ID to filter results.
            n_results: Number of results to return.

        Returns:
            List of matching documents with metadata.
        """
        collection = self._get_or_create_collection("resumes")

        query_embedding = self.embeddings.embed_query(query)

        where_filter = None
        if resume_id:
            where_filter = {"resume_id": resume_id}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                    }
                )

        return search_results

    async def index_job_description(
        self,
        job_id: str,
        title: str,
        company: str,
        description: str,
        requirements: str | None = None,
    ) -> None:
        """
        Index a job description for matching.

        Args:
            job_id: Unique identifier for the job.
            title: Job title.
            company: Company name.
            description: Full job description.
            requirements: Optional requirements section.
        """
        collection = self._get_or_create_collection("job_descriptions")

        full_text = f"{title}\n\n{description}"
        if requirements:
            full_text += f"\n\nRequirements:\n{requirements}"

        embedding = self.embeddings.embed_query(full_text)

        collection.add(
            documents=[full_text],
            embeddings=[embedding],
            metadatas=[
                {
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                }
            ],
            ids=[job_id],
        )

    async def find_similar_jobs(
        self, resume_text: str, n_results: int = 5
    ) -> list[dict]:
        """
        Find jobs similar to a resume.

        Args:
            resume_text: Resume text to match against.
            n_results: Number of results to return.

        Returns:
            List of matching job descriptions.
        """
        collection = self._get_or_create_collection("job_descriptions")

        query_embedding = self.embeddings.embed_query(resume_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        job_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                job_results.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "similarity": 1
                        - (results["distances"][0][i] if results["distances"] else 0),
                    }
                )

        return job_results

    async def delete_resume_index(self, resume_id: str) -> None:
        """
        Delete all indexed content for a resume.

        Args:
            resume_id: ID of the resume to delete.
        """
        collection = self._get_or_create_collection("resumes")

        results = collection.get(where={"resume_id": resume_id}, include=["metadatas"])

        if results["ids"]:
            collection.delete(ids=results["ids"])

    async def index_company_info(self, company_name: str, info: dict) -> None:
        """
        Index company information for research.

        Args:
            company_name: Name of the company.
            info: Dictionary containing company information.
        """
        collection = self._get_or_create_collection("companies")

        content_parts = [f"Company: {company_name}"]
        for key, value in info.items():
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                content_parts.append(f"{key}: {', '.join(str(v) for v in value)}")

        full_content = "\n".join(content_parts)
        embedding = self.embeddings.embed_query(full_content)

        # Convert list values to comma-separated strings for ChromaDB metadata
        clean_metadata = {"company_name": company_name}
        for key, value in info.items():
            if isinstance(value, list):
                clean_metadata[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)

        collection.upsert(
            documents=[full_content],
            embeddings=[embedding],
            metadatas=[clean_metadata],
            ids=[company_name.lower().replace(" ", "_")],
        )

    async def search_company_info(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Search for company information.

        Args:
            query: Search query.
            n_results: Number of results to return.

        Returns:
            List of matching company information.
        """
        collection = self._get_or_create_collection("companies")

        query_embedding = self.embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        company_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                company_results.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                    }
                )

        return company_results
