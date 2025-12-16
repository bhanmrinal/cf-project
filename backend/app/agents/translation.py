"""
Translation & Localization Agent.

Translates resumes while maintaining professional quality and adapts
formatting and content for different cultural contexts.
"""

import re
from typing import Any, Optional

from backend.app.agents.base import BaseAgent, AgentResult
from backend.app.models.resume import Resume
from backend.app.models.conversation import Conversation, AgentType


class TranslationAgent(BaseAgent):
    """
    Agent specialized in resume translation and localization.

    Capabilities:
    - Translate resumes to different languages
    - Adapt content for cultural contexts
    - Adjust formatting for regional conventions
    - Research local hiring practices
    - Maintain professional tone in translations
    """

    agent_type = AgentType.TRANSLATION
    description = "Translates and localizes resumes for different markets"

    SUPPORTED_LANGUAGES = {
        "spanish": {
            "code": "es",
            "regions": ["Spain", "Mexico", "Argentina", "Colombia"],
        },
        "french": {
            "code": "fr",
            "regions": ["France", "Canada", "Belgium", "Switzerland"],
        },
        "german": {"code": "de", "regions": ["Germany", "Austria", "Switzerland"]},
        "portuguese": {"code": "pt", "regions": ["Brazil", "Portugal"]},
        "italian": {"code": "it", "regions": ["Italy", "Switzerland"]},
        "dutch": {"code": "nl", "regions": ["Netherlands", "Belgium"]},
        "japanese": {"code": "ja", "regions": ["Japan"]},
        "chinese": {"code": "zh", "regions": ["China", "Taiwan", "Singapore"]},
        "korean": {"code": "ko", "regions": ["South Korea"]},
        "arabic": {"code": "ar", "regions": ["UAE", "Saudi Arabia", "Egypt"]},
        "hindi": {"code": "hi", "regions": ["India"]},
        "russian": {"code": "ru", "regions": ["Russia"]},
    }

    REGIONAL_CONVENTIONS = {
        "Germany": {
            "photo": "Often expected",
            "personal_info": "Date of birth, nationality common",
            "format": "Reverse chronological, detailed",
            "length": "2-3 pages acceptable",
            "notes": "Formal tone, include all certifications",
        },
        "France": {
            "photo": "Common but not required",
            "personal_info": "Age, marital status sometimes included",
            "format": "Reverse chronological",
            "length": "1-2 pages",
            "notes": "Include language proficiency levels",
        },
        "Japan": {
            "photo": "Required",
            "personal_info": "Date of birth, gender expected",
            "format": "Specific rirekisho format often required",
            "length": "1-2 pages",
            "notes": "Very formal, humble tone",
        },
        "Spain": {
            "photo": "Common",
            "personal_info": "DNI number sometimes included",
            "format": "Europass format accepted",
            "length": "1-2 pages",
            "notes": "Include language certifications",
        },
        "Mexico": {
            "photo": "Often expected",
            "personal_info": "CURP sometimes included",
            "format": "Similar to US but more personal info",
            "length": "1-2 pages",
            "notes": "Professional Spanish, formal tone",
        },
        "Brazil": {
            "photo": "Common",
            "personal_info": "CPF sometimes included",
            "format": "Similar to US",
            "length": "1-2 pages",
            "notes": "Portuguese (Brazilian), include courses/certifications",
        },
        "India": {
            "photo": "Common",
            "personal_info": "Father's name sometimes included",
            "format": "Detailed, comprehensive",
            "length": "2-3 pages acceptable",
            "notes": "Include all educational details",
        },
        "UAE": {
            "photo": "Expected",
            "personal_info": "Nationality, visa status important",
            "format": "Comprehensive",
            "length": "2+ pages acceptable",
            "notes": "Include nationality and visa status",
        },
    }

    def __init__(self, temperature: float = 0.3):
        super().__init__(temperature)

    def get_system_prompt(self) -> str:
        return """You are an expert professional translator and international career consultant.

Your role is to:
1. Translate resumes accurately while maintaining professional quality
2. Adapt content for specific cultural and regional contexts
3. Apply local resume formatting conventions
4. Ensure industry-specific terminology is correctly translated
5. Maintain the candidate's qualifications and achievements accurately

Translation Guidelines:
- Use formal, professional language appropriate for the target region
- Preserve technical terms that are commonly used in English (e.g., "software engineer")
- Adapt date formats, address formats to local conventions
- Translate job titles to their local equivalents where appropriate
- Maintain action verbs and achievement-focused language

Cultural Adaptation:
- Adjust the level of personal information based on regional norms
- Modify the resume structure if local conventions differ
- Include region-specific sections if needed (e.g., photo placeholder for Germany)
- Adapt the tone to match local professional communication styles

Output Format:
- Provide the translated resume with "## Section Name" headers (in the target language)
- Include a brief note about cultural adaptations made
- Highlight any terms kept in English and why"""

    async def process(
        self,
        user_message: str,
        resume: Resume,
        conversation: Conversation,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Process a translation/localization request.

        Args:
            user_message: User's request (e.g., "Translate to Spanish for Mexico")
            resume: Current resume to translate
            conversation: Conversation context
            context: Additional context including target language/region

        Returns:
            AgentResult with translated resume
        """
        target_language = context.get("target_language") or self._extract_language(
            user_message
        )
        target_region = context.get("target_region") or self._extract_region(
            user_message
        )

        if not target_language:
            return AgentResult(
                success=False,
                message=self._get_language_help_message(),
                reasoning="No target language identified",
            )

        language_info = self.SUPPORTED_LANGUAGES.get(target_language.lower())
        if not language_info:
            return AgentResult(
                success=False,
                message=f"I don't currently support translation to '{target_language}'. {self._get_language_help_message()}",
                reasoning=f"Unsupported language: {target_language}",
            )

        if not target_region:
            target_region = language_info["regions"][0]

        regional_conventions = self.REGIONAL_CONVENTIONS.get(target_region, {})

        translation_prompt = self._build_translation_prompt(
            resume=resume,
            target_language=target_language,
            target_region=target_region,
            regional_conventions=regional_conventions,
            user_message=user_message,
        )

        response = await self._invoke_llm(
            system_prompt=self.get_system_prompt(), user_prompt=translation_prompt
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
                "translated_to": target_language,
                "target_region": target_region,
                "optimization_type": "translation",
            },
        )

        cultural_notes = self._extract_cultural_notes(response)

        message = f"""🌍 **Translation Complete**

**Target Language:** {target_language.title()}
**Target Region:** {target_region}

**Regional Conventions Applied:**
{self._format_conventions(regional_conventions)}

**Cultural Adaptations:**
{cultural_notes}

Your resume has been translated and localized. Review the changes below:"""

        return AgentResult(
            success=True,
            message=message,
            updated_resume=updated_resume,
            updated_sections=updated_sections,
            changes=changes,
            reasoning=cultural_notes,
            metadata={
                "target_language": target_language,
                "target_region": target_region,
                "regional_conventions": regional_conventions,
            },
        )

    def _extract_language(self, message: str) -> Optional[str]:
        """Extract target language from user message."""
        message_lower = message.lower()

        for language in self.SUPPORTED_LANGUAGES.keys():
            if language in message_lower:
                return language

        language_patterns = [
            r"(?:translate|convert|change).*?(?:to|into)\s+(\w+)",
            r"(\w+)\s+(?:version|translation|resume)",
            r"in\s+(\w+)(?:\s+language)?",
        ]

        for pattern in language_patterns:
            match = re.search(pattern, message_lower)
            if match:
                potential_lang = match.group(1)
                if potential_lang in self.SUPPORTED_LANGUAGES:
                    return potential_lang

        return None

    def _extract_region(self, message: str) -> Optional[str]:
        """Extract target region from user message."""
        message_lower = message.lower()

        all_regions = []
        for lang_info in self.SUPPORTED_LANGUAGES.values():
            all_regions.extend(lang_info["regions"])

        for region in all_regions:
            if region.lower() in message_lower:
                return region

        market_patterns = [
            r"(?:for|targeting|in)\s+(?:the\s+)?(\w+)\s+market",
            r"(\w+)\s+(?:market|region|country)",
        ]

        for pattern in market_patterns:
            match = re.search(pattern, message_lower)
            if match:
                potential_region = match.group(1).title()
                if potential_region in all_regions:
                    return potential_region

        return None

    def _get_language_help_message(self) -> str:
        """Get help message listing supported languages."""
        languages = ", ".join(lang.title() for lang in self.SUPPORTED_LANGUAGES.keys())
        return f"""Please specify which language you'd like me to translate your resume to.

**Supported Languages:**
{languages}

**Example requests:**
- "Translate my resume to Spanish for the Mexican market"
- "Convert to German"
- "Create a French version for Canada"
- "Translate to Japanese"

You can also specify a region for more accurate localization."""

    def _build_translation_prompt(
        self,
        resume: Resume,
        target_language: str,
        target_region: str,
        regional_conventions: dict[str, str],
        user_message: str,
    ) -> str:
        """Build the translation prompt for the LLM."""
        resume_content = self._format_resume_for_prompt(resume)

        conventions_text = (
            "\n".join(
                f"- {key}: {value}" for key, value in regional_conventions.items()
            )
            if regional_conventions
            else "Standard international format"
        )

        return f"""User Request: {user_message}

Target Language: {target_language.title()}
Target Region/Market: {target_region}

Regional Resume Conventions for {target_region}:
{conventions_text}

Original Resume (English):
{resume_content}

Please translate and localize this resume for the {target_region} market:

1. Translate all content to {target_language.title()}
2. Apply regional formatting conventions
3. Adapt section headers to local standards
4. Translate job titles appropriately (keep technical terms in English if commonly used)
5. Adjust date formats to local conventions
6. Modify personal information section based on regional norms

Provide the translated resume with section headers in {target_language.title()} using "## Section Name" format.

At the end, include a brief note titled "CULTURAL_NOTES:" explaining:
- Key cultural adaptations made
- Any terms kept in English and why
- Suggestions for additional localization (e.g., adding photo for German market)

IMPORTANT: Maintain accuracy of qualifications and achievements. Do not embellish or change factual content."""

    def _extract_cultural_notes(self, response: str) -> str:
        """Extract cultural notes from the LLM response."""
        notes_match = re.search(
            r"CULTURAL_NOTES?:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE
        )

        if notes_match:
            return notes_match.group(1).strip()[:500]

        last_paragraph = response.strip().split("\n\n")[-1]
        if len(last_paragraph) < 500:
            return last_paragraph

        return "Resume translated and adapted for the target market."

    def _format_conventions(self, conventions: dict[str, str]) -> str:
        """Format regional conventions for display."""
        if not conventions:
            return "• Standard international format applied"

        return "\n".join(
            f"• **{key.replace('_', ' ').title()}:** {value}"
            for key, value in conventions.items()
        )

    async def get_localization_suggestions(
        self, resume: Resume, target_region: str
    ) -> list[str]:
        """
        Get suggestions for localizing a resume to a specific region.

        Args:
            resume: The resume to analyze.
            target_region: The target region.

        Returns:
            List of localization suggestions.
        """
        conventions = self.REGIONAL_CONVENTIONS.get(target_region, {})
        suggestions = []

        if conventions.get("photo") in ["Required", "Often expected", "Common"]:
            suggestions.append(
                f"Consider adding a professional photo (common in {target_region})"
            )

        if "personal_info" in conventions:
            suggestions.append(
                f"Personal info expectations: {conventions['personal_info']}"
            )

        if conventions.get("length"):
            suggestions.append(f"Recommended length: {conventions['length']}")

        if conventions.get("notes"):
            suggestions.append(conventions["notes"])

        return suggestions
