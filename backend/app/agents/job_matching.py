"""
Job Description Matching Agent.

Analyzes job descriptions, matches them against resumes, and restructures
resume content to highlight relevant experience and identify skill gaps.
"""

import re
from typing import Any, Optional

from backend.app.agents.base import BaseAgent, AgentResult
from backend.app.models.resume import Resume
from backend.app.models.conversation import Conversation, AgentType
from backend.app.services.vector_store import VectorStoreService


class JobMatchingAgent(BaseAgent):
    """
    Agent specialized in job description analysis and resume matching.

    Capabilities:
    - Analyze job descriptions to extract requirements
    - Calculate match scores between resumes and jobs
    - Identify skill gaps and missing qualifications
    - Restructure resumes to highlight relevant experience
    - Suggest improvements to increase match percentage
    """

    agent_type = AgentType.JOB_MATCHING
    description = "Analyzes job descriptions and optimizes resumes for better matching"

    def __init__(self, temperature: float = 0.5):
        super().__init__(temperature)
        self.vector_store = VectorStoreService()

    def get_system_prompt(self) -> str:
        return """You are an expert ATS (Applicant Tracking System) specialist and career coach.

Your role is to:
1. Analyze job descriptions to identify key requirements, skills, and qualifications
2. Compare resumes against job requirements to calculate match scores
3. Identify gaps between resume content and job requirements
4. Restructure and optimize resumes to better match specific job descriptions
5. Provide actionable recommendations to improve match rates

When analyzing and optimizing:
- Extract both explicit requirements and implied preferences from job descriptions
- Consider both hard skills (technical) and soft skills
- Identify keywords that ATS systems would look for
- Prioritize changes that have the highest impact on match scores
- Maintain truthfulness - suggest highlighting existing skills, not fabricating new ones

Output Format:
- Provide match scores as percentages with breakdowns
- List skill gaps clearly with suggestions
- Format optimized resumes with "## Section Name" headers
- Always explain your analysis and recommendations"""

    async def process(
        self,
        user_message: str,
        resume: Resume,
        conversation: Conversation,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Process a job matching request.

        Args:
            user_message: User's request with job description
            resume: Current resume to match/optimize
            conversation: Conversation context
            context: Additional context including job description

        Returns:
            AgentResult with analysis, match score, and optimized resume
        """
        job_description = context.get(
            "job_description"
        ) or self._extract_job_description(user_message)

        if not job_description:
            return AgentResult(
                success=False,
                message="I need a job description to analyze. Please provide the job description you'd like me to match your resume against.",
                reasoning="No job description found in request or context",
            )

        job_analysis = await self._analyze_job_description(job_description)

        match_result = await self._calculate_match_score(resume, job_analysis)

        optimization_prompt = self._build_optimization_prompt(
            resume=resume,
            job_description=job_description,
            job_analysis=job_analysis,
            match_result=match_result,
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
                "job_match_score": match_result["overall_score"],
                "optimization_type": "job_matching",
            },
        )

        message = self._format_match_message(match_result, job_analysis)

        return AgentResult(
            success=True,
            message=message,
            updated_resume=updated_resume,
            updated_sections=updated_sections,
            changes=changes,
            reasoning=self._extract_reasoning(response),
            metadata={
                "job_analysis": job_analysis,
                "match_result": match_result,
                "original_score": match_result["overall_score"],
            },
        )

    def _extract_job_description(self, message: str) -> Optional[str]:
        """Extract job description from user message."""
        jd_indicators = [
            "job description:",
            "jd:",
            "position:",
            "role:",
            "responsibilities:",
            "requirements:",
            "qualifications:",
        ]

        message_lower = message.lower()
        for indicator in jd_indicators:
            if indicator in message_lower:
                idx = message_lower.find(indicator)
                return message[idx:].strip()

        if len(message) > 200 and any(
            word in message_lower
            for word in [
                "responsibilities",
                "requirements",
                "qualifications",
                "experience",
                "we are looking",
                "you will",
                "must have",
                "years of experience",
            ]
        ):
            return message

        return None

    async def _analyze_job_description(self, job_description: str) -> dict[str, Any]:
        """
        Analyze a job description to extract requirements.

        Args:
            job_description: The job description text.

        Returns:
            Dictionary with analyzed requirements.
        """
        analysis_prompt = f"""Analyze the following job description and extract:

Job Description:
{job_description}

Please extract and categorize:
1. REQUIRED_SKILLS: Technical/hard skills that are required (comma-separated)
2. PREFERRED_SKILLS: Skills that are preferred but not required (comma-separated)
3. SOFT_SKILLS: Soft skills and qualities mentioned (comma-separated)
4. EXPERIENCE_YEARS: Minimum years of experience required (number or "Not specified")
5. EDUCATION: Education requirements
6. KEY_RESPONSIBILITIES: Main job responsibilities (bullet points)
7. KEYWORDS: Important keywords for ATS matching (comma-separated)
8. COMPANY_VALUES: Any company values or culture indicators mentioned

Format your response exactly as:
REQUIRED_SKILLS: skill1, skill2, skill3
PREFERRED_SKILLS: skill1, skill2
SOFT_SKILLS: skill1, skill2
EXPERIENCE_YEARS: X
EDUCATION: requirement
KEY_RESPONSIBILITIES:
- responsibility 1
- responsibility 2
KEYWORDS: keyword1, keyword2, keyword3
COMPANY_VALUES: value1, value2"""

        response = await self._invoke_llm(
            system_prompt="You are a job description analyst. Extract information precisely and concisely.",
            user_prompt=analysis_prompt,
        )

        return self._parse_job_analysis(response)

    def _parse_job_analysis(self, response: str) -> dict[str, Any]:
        """Parse job analysis from LLM response."""
        analysis = {
            "required_skills": [],
            "preferred_skills": [],
            "soft_skills": [],
            "experience_years": None,
            "education": "",
            "key_responsibilities": [],
            "keywords": [],
            "company_values": [],
        }

        patterns = {
            "required_skills": r"REQUIRED_SKILLS:\s*(.+?)(?=PREFERRED_SKILLS:|$)",
            "preferred_skills": r"PREFERRED_SKILLS:\s*(.+?)(?=SOFT_SKILLS:|$)",
            "soft_skills": r"SOFT_SKILLS:\s*(.+?)(?=EXPERIENCE_YEARS:|$)",
            "experience_years": r"EXPERIENCE_YEARS:\s*(.+?)(?=EDUCATION:|$)",
            "education": r"EDUCATION:\s*(.+?)(?=KEY_RESPONSIBILITIES:|$)",
            "keywords": r"KEYWORDS:\s*(.+?)(?=COMPANY_VALUES:|$)",
            "company_values": r"COMPANY_VALUES:\s*(.+?)$",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in [
                    "required_skills",
                    "preferred_skills",
                    "soft_skills",
                    "keywords",
                    "company_values",
                ]:
                    analysis[key] = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "experience_years":
                    years_match = re.search(r"(\d+)", value)
                    analysis[key] = int(years_match.group(1)) if years_match else None
                else:
                    analysis[key] = value

        resp_match = re.search(
            r"KEY_RESPONSIBILITIES:\s*(.+?)(?=KEYWORDS:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if resp_match:
            resp_text = resp_match.group(1)
            analysis["key_responsibilities"] = [
                line.strip().lstrip("-•").strip()
                for line in resp_text.split("\n")
                if line.strip() and line.strip() not in ["-", "•"]
            ]

        return analysis

    async def _calculate_match_score(
        self, resume: Resume, job_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate match score between resume and job requirements.

        Args:
            resume: The resume to evaluate.
            job_analysis: Analyzed job requirements.

        Returns:
            Dictionary with match scores and details.
        """
        resume_text = self._format_resume_for_prompt(resume).lower()

        def count_matches(skills: list[str]) -> tuple[list[str], list[str]]:
            found = []
            missing = []
            for skill in skills:
                skill_lower = skill.lower()
                skill_variants = [
                    skill_lower,
                    skill_lower.replace("-", " "),
                    skill_lower.replace(" ", "-"),
                ]
                if any(variant in resume_text for variant in skill_variants):
                    found.append(skill)
                else:
                    missing.append(skill)
            return found, missing

        required_found, required_missing = count_matches(
            job_analysis["required_skills"]
        )
        preferred_found, preferred_missing = count_matches(
            job_analysis["preferred_skills"]
        )
        soft_found, soft_missing = count_matches(job_analysis["soft_skills"])
        keywords_found, keywords_missing = count_matches(job_analysis["keywords"])

        required_total = len(job_analysis["required_skills"]) or 1
        preferred_total = len(job_analysis["preferred_skills"]) or 1
        soft_total = len(job_analysis["soft_skills"]) or 1
        keywords_total = len(job_analysis["keywords"]) or 1

        required_score = (len(required_found) / required_total) * 100
        preferred_score = (len(preferred_found) / preferred_total) * 100
        soft_score = (len(soft_found) / soft_total) * 100
        keywords_score = (len(keywords_found) / keywords_total) * 100

        overall_score = (
            required_score * 0.4
            + preferred_score * 0.2
            + soft_score * 0.15
            + keywords_score * 0.25
        )

        return {
            "overall_score": round(overall_score, 1),
            "required_skills": {
                "score": round(required_score, 1),
                "found": required_found,
                "missing": required_missing,
            },
            "preferred_skills": {
                "score": round(preferred_score, 1),
                "found": preferred_found,
                "missing": preferred_missing,
            },
            "soft_skills": {
                "score": round(soft_score, 1),
                "found": soft_found,
                "missing": soft_missing,
            },
            "keywords": {
                "score": round(keywords_score, 1),
                "found": keywords_found,
                "missing": keywords_missing,
            },
            "skill_gaps": required_missing + preferred_missing[:3],
            "recommendations": self._generate_recommendations(
                required_missing, preferred_missing, soft_missing
            ),
        }

    def _generate_recommendations(
        self,
        required_missing: list[str],
        preferred_missing: list[str],
        soft_missing: list[str],
    ) -> list[str]:
        """Generate recommendations based on skill gaps."""
        recommendations = []

        if required_missing:
            recommendations.append(
                f"Critical: Add or highlight experience with: {', '.join(required_missing[:5])}"
            )

        if preferred_missing:
            recommendations.append(
                f"Recommended: Consider adding: {', '.join(preferred_missing[:3])}"
            )

        if soft_missing:
            recommendations.append(
                f"Soft skills to demonstrate: {', '.join(soft_missing[:3])}"
            )

        if not recommendations:
            recommendations.append("Your resume is well-aligned with this job!")

        return recommendations

    def _build_optimization_prompt(
        self,
        resume: Resume,
        job_description: str,
        job_analysis: dict[str, Any],
        match_result: dict[str, Any],
        user_message: str,
    ) -> str:
        """Build the optimization prompt for the LLM."""
        resume_content = self._format_resume_for_prompt(resume)

        return f"""User Request: {user_message}

Job Description:
{job_description[:2000]}

Job Analysis:
- Required Skills: {", ".join(job_analysis["required_skills"])}
- Preferred Skills: {", ".join(job_analysis["preferred_skills"])}
- Soft Skills: {", ".join(job_analysis["soft_skills"])}
- Key Responsibilities: {"; ".join(job_analysis["key_responsibilities"][:5])}
- Important Keywords: {", ".join(job_analysis["keywords"])}

Current Match Score: {match_result["overall_score"]}%
- Required Skills Match: {match_result["required_skills"]["score"]}%
- Missing Required Skills: {", ".join(match_result["required_skills"]["missing"])}
- Missing Keywords: {", ".join(match_result["keywords"]["missing"][:10])}

Current Resume:
{resume_content}

Please optimize this resume to better match the job description:
1. Incorporate missing keywords naturally where the candidate has relevant experience
2. Restructure bullet points to emphasize relevant responsibilities
3. Highlight transferable skills that match the requirements
4. Ensure the summary/objective aligns with the role
5. Use action verbs and quantifiable achievements where possible

Provide the optimized resume with clear section headers (## Section Name).
At the end, explain the key changes made and the expected improvement in match score.

IMPORTANT: Only add skills/keywords where the candidate has genuine experience. Do not fabricate qualifications."""

    def _format_match_message(
        self, match_result: dict[str, Any], job_analysis: dict[str, Any]
    ) -> str:
        """Format the match result message."""
        score = match_result["overall_score"]
        skill_gaps = match_result["skill_gaps"]

        if score >= 80:
            rating = "Excellent match!"
        elif score >= 60:
            rating = "Good match with room for improvement"
        elif score >= 40:
            rating = "Moderate match - optimization recommended"
        else:
            rating = "Low match - significant optimization needed"

        message = f"""📊 **Match Analysis Complete**

**Overall Match Score: {score}%** - {rating}

**Score Breakdown:**
- Required Skills: {match_result["required_skills"]["score"]}%
- Preferred Skills: {match_result["preferred_skills"]["score"]}%
- Soft Skills: {match_result["soft_skills"]["score"]}%
- Keywords: {match_result["keywords"]["score"]}%

**Skill Gaps Identified:**
{chr(10).join(f"• {gap}" for gap in skill_gaps[:5]) if skill_gaps else "• None - great job!"}

**Recommendations:**
{chr(10).join(f"• {rec}" for rec in match_result["recommendations"])}

I've optimized your resume to better match this position. See the changes below:"""

        return message

    def _extract_reasoning(self, response: str) -> str:
        """Extract the reasoning/explanation from the LLM response."""
        reasoning_patterns = [
            r"(?:Key Changes|Changes Made|Improvements|Optimization Summary):\s*(.+?)$",
            r"(?:Expected improvement|This should|These changes)(.+?)$",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]

        return "Resume optimized to better match the job requirements."
