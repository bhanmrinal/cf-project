"""
Company Research & Optimization Agent.

Researches target companies using web APIs and optimizes resume content
to match company culture, values, and hiring patterns.
"""

from typing import Any

from app.agents.base import AgentResult, BaseAgent
from app.models.conversation import AgentType, Conversation
from app.models.resume import Resume
from app.services.vector_store import VectorStoreService


class CompanyResearchAgent(BaseAgent):
    """
    Agent specialized in company research and resume optimization.

    Capabilities:
    - Research company culture, values, and mission
    - Analyze company hiring patterns and preferences
    - Optimize resume language to match company tone
    - Highlight relevant experience for the target company
    """

    agent_type = AgentType.COMPANY_RESEARCH
    description = (
        "Researches companies and optimizes resumes to match their culture and values"
    )

    def __init__(self, temperature: float = 0.7):
        super().__init__(temperature)
        self.vector_store = VectorStoreService()

    def get_system_prompt(self) -> str:
        return """You are an expert career consultant and resume optimizer specializing in company research.

Your role is to:
1. Research and understand target companies (culture, values, mission, hiring patterns)
2. Optimize resumes to align with specific company preferences
3. Adjust language, tone, and emphasis to match company culture
4. Highlight experiences and skills most relevant to the target company

When optimizing a resume:
- Maintain truthfulness - never fabricate experiences or skills
- Use language and keywords that resonate with the company's values
- Emphasize achievements that align with the company's mission
- Structure content to highlight the most relevant qualifications first
- Keep the professional tone consistent with the company's culture

Output Format:
When providing an optimized resume, format it with clear section headers using "## Section Name" format.
Always explain your reasoning and the specific changes made."""

    async def process(
        self,
        user_message: str,
        resume: Resume,
        conversation: Conversation,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Process a company research and optimization request.

        Args:
            user_message: User's request (e.g., "Optimize my resume for Google")
            resume: Current resume to optimize
            conversation: Conversation context
            context: Additional context including target company

        Returns:
            AgentResult with optimized resume and reasoning
        """
        target_company = context.get("target_company") or self._extract_company_name(
            user_message
        )

        if not target_company:
            return AgentResult(
                success=False,
                message="I couldn't identify a target company. Please specify which company you'd like me to optimize your resume for.",
                reasoning="No company name found in request or context",
            )

        company_info = await self._research_company(target_company)

        optimization_prompt = self._build_optimization_prompt(
            resume=resume,
            company_name=target_company,
            company_info=company_info,
            user_message=user_message,
        )

        response = await self._invoke_llm(
            system_prompt=self.get_system_prompt(), user_prompt=optimization_prompt
        )

        updated_sections = self._extract_sections_from_response(response, resume)
        changes = self._identify_changes(resume.sections, updated_sections)

        updated_resume = Resume(
            id=resume.id,
            user_id=resume.user_id,
            filename=resume.filename,
            raw_text=resume.raw_text,
            sections=updated_sections,
            metadata={
                **resume.metadata,
                "optimized_for": target_company,
                "optimization_type": "company_research",
            },
        )

        reasoning = self._extract_reasoning(response)

        return AgentResult(
            success=True,
            message=f"I've optimized your resume for {target_company}. Here's what I changed and why:",
            updated_resume=updated_resume,
            updated_sections=updated_sections,
            changes=changes,
            reasoning=reasoning,
            metadata={"target_company": target_company, "company_info": company_info},
        )

    def _extract_company_name(self, message: str) -> str | None:
        """Extract company name from user message."""
        import re

        patterns = [
            r"(?:for|at|to)\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s*$|\s*\.|\s*,|\s+resume|\s+job|\s+position)",
            r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+(?:company|corporation|inc|corp|ltd)",
            r"optimize.*?(?:for|to)\s+([A-Z][A-Za-z0-9\s&]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        words = message.split()
        for word in words:
            if (
                word[0].isupper()
                and len(word) > 2
                and word.lower()
                not in {
                    "optimize",
                    "resume",
                    "the",
                    "for",
                    "and",
                    "with",
                    "make",
                    "update",
                }
            ):
                return word

        return None

    async def _research_company(self, company_name: str) -> dict[str, Any]:
        """
        Research a company using web search and cached data.

        Args:
            company_name: Name of the company to research.

        Returns:
            Dictionary containing company information.
        """
        cached_info = await self.vector_store.search_company_info(
            company_name, n_results=1
        )
        if (
            cached_info
            and cached_info[0].get("metadata", {}).get("company_name", "").lower()
            == company_name.lower()
        ):
            return cached_info[0].get("metadata", {})

        company_info = await self._web_search_company(company_name)

        if company_info:
            await self.vector_store.index_company_info(company_name, company_info)

        return company_info

    async def _web_search_company(self, company_name: str) -> dict[str, Any]:
        """
        Search the web for company information.

        Args:
            company_name: Name of the company.

        Returns:
            Dictionary with company information.
        """
        try:
            # Try new package name first, fall back to old name
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = list(
                ddgs.text(
                    f"{company_name} company culture values mission", max_results=5
                )
            )

            culture_info = " ".join([r.get("body", "") for r in results[:3]])

            hiring_results = list(
                ddgs.text(
                    f"{company_name} hiring process interview what they look for",
                    max_results=3,
                )
            )

            hiring_info = " ".join([r.get("body", "") for r in hiring_results[:2]])

            summary_prompt = f"""Based on the following information about {company_name}, extract key details:

Culture and Values Information:
{culture_info[:2000]}

Hiring Information:
{hiring_info[:1000]}

Please extract and summarize:
1. Company culture and values (2-3 sentences)
2. Key skills and qualities they look for
3. Industry and main business areas
4. Any notable hiring preferences or patterns

Format your response as:
CULTURE: <summary>
KEY_SKILLS: <comma-separated list>
INDUSTRY: <industry>
HIRING_NOTES: <notes>"""

            response = await self._invoke_llm(
                system_prompt="You are a company research analyst. Extract and summarize company information concisely.",
                user_prompt=summary_prompt,
            )

            return self._parse_company_info(response, company_name)

        except Exception as e:
            return {
                "company_name": company_name,
                "culture": "Information not available - using general best practices",
                "key_skills": [],
                "industry": "Unknown",
                "hiring_notes": "No specific information found",
                "error": str(e),
            }

    def _parse_company_info(self, response: str, company_name: str) -> dict[str, Any]:
        """Parse company info from LLM response."""
        import re

        info = {
            "company_name": company_name,
            "culture": "",
            "key_skills": [],
            "industry": "",
            "hiring_notes": "",
        }

        culture_match = re.search(
            r"CULTURE:\s*(.+?)(?=KEY_SKILLS:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if culture_match:
            info["culture"] = culture_match.group(1).strip()

        skills_match = re.search(
            r"KEY_SKILLS:\s*(.+?)(?=INDUSTRY:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if skills_match:
            skills_text = skills_match.group(1).strip()
            info["key_skills"] = [
                s.strip() for s in skills_text.split(",") if s.strip()
            ]

        industry_match = re.search(
            r"INDUSTRY:\s*(.+?)(?=HIRING_NOTES:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if industry_match:
            info["industry"] = industry_match.group(1).strip()

        hiring_match = re.search(
            r"HIRING_NOTES:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE
        )
        if hiring_match:
            info["hiring_notes"] = hiring_match.group(1).strip()

        return info

    def _build_optimization_prompt(
        self,
        resume: Resume,
        company_name: str,
        company_info: dict[str, Any],
        user_message: str,
    ) -> str:
        """Build the optimization prompt for the LLM."""
        resume_content = self._format_resume_for_prompt(resume)

        return f"""User Request: {user_message}

Target Company: {company_name}

Company Information:
- Culture & Values: {company_info.get("culture", "Not available")}
- Key Skills They Look For: {", ".join(company_info.get("key_skills", ["Not available"]))}
- Industry: {company_info.get("industry", "Not available")}
- Hiring Notes: {company_info.get("hiring_notes", "Not available")}

Current Resume:
{resume_content}

Please optimize this resume for {company_name}. Make the following improvements:
1. Adjust the language and tone to match the company's culture
2. Highlight experiences and achievements most relevant to their industry
3. Incorporate keywords and skills they value
4. Ensure the summary/objective aligns with their mission
5. Reorder or emphasize sections to best match what they're looking for

Provide the optimized resume with clear section headers (## Section Name) and explain your key changes at the end.

IMPORTANT: Only modify content that benefits the application. Maintain truthfulness and don't fabricate experiences."""

    def _extract_reasoning(self, response: str) -> str:
        """Extract the reasoning/explanation from the LLM response."""
        import re

        reasoning_patterns = [
            r"(?:Key Changes|Changes Made|Reasoning|Explanation|What I Changed):\s*(.+?)$",
            r"(?:Here's what I|I have|I've).*?(?:changed|modified|updated|optimized)(.+?)$",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]

        lines = response.strip().split("\n")
        last_paragraph = []
        for line in reversed(lines):
            if line.strip():
                last_paragraph.insert(0, line)
            elif last_paragraph:
                break

        return (
            " ".join(last_paragraph)[:500]
            if last_paragraph
            else "Resume optimized for target company."
        )
