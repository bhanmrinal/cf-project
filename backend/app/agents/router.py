"""
Conversation Router.

Intelligently routes user messages to the appropriate specialized agent
based on intent classification and conversation context.
"""

import re
from typing import Any, Optional

from backend.app.core.llm import get_llm
from backend.app.models.conversation import Conversation, AgentType
from backend.app.models.resume import Resume
from backend.app.agents.base import BaseAgent, AgentResult
from backend.app.agents.company_research import CompanyResearchAgent
from backend.app.agents.job_matching import JobMatchingAgent
from backend.app.agents.translation import TranslationAgent


class ConversationRouter:
    """
    Routes conversations to appropriate specialized agents.

    Responsibilities:
    - Classify user intent from messages
    - Maintain conversation context
    - Route to appropriate agent
    - Handle multi-agent workflows
    - Manage fallback responses
    """

    INTENT_PATTERNS = {
        AgentType.COMPANY_RESEARCH: [
            r"optimize.*(?:for|at)\s+\w+",
            r"(?:target|apply|applying).*company",
            r"(?:google|amazon|microsoft|meta|apple|netflix|spotify|uber|airbnb)",
            r"company\s+(?:culture|values|research)",
            r"tailor.*(?:for|to)\s+\w+",
        ],
        AgentType.JOB_MATCHING: [
            r"job\s+description",
            r"match.*(?:job|position|role)",
            r"(?:jd|job desc)",
            r"skill\s+gap",
            r"match\s+score",
            r"requirements",
            r"fit.*(?:job|position|role)",
            r"ats",
        ],
        AgentType.TRANSLATION: [
            r"translat",
            r"(?:spanish|french|german|portuguese|italian|japanese|chinese|korean|arabic|hindi|russian|dutch)",
            r"(?:spain|mexico|france|germany|brazil|japan|china|india)",
            r"locali[sz]",
            r"(?:foreign|international)\s+market",
            r"(?:different|another)\s+language",
        ],
    }

    def __init__(self):
        self._llm = None
        self._agents: dict[AgentType, BaseAgent] = {}

    @property
    def llm(self):
        """Lazy initialization of LLM for intent classification."""
        if self._llm is None:
            self._llm = get_llm(temperature=0.1)
        return self._llm

    def _get_agent(self, agent_type: AgentType) -> BaseAgent:
        """Get or create an agent instance."""
        if agent_type not in self._agents:
            agent_map = {
                AgentType.COMPANY_RESEARCH: CompanyResearchAgent,
                AgentType.JOB_MATCHING: JobMatchingAgent,
                AgentType.TRANSLATION: TranslationAgent,
            }
            agent_class = agent_map.get(agent_type)
            if agent_class:
                self._agents[agent_type] = agent_class()
        return self._agents.get(agent_type)

    async def route(
        self,
        user_message: str,
        resume: Optional[Resume],
        conversation: Conversation,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Route a user message to the appropriate agent.

        Args:
            user_message: The user's message.
            resume: The current resume (if any).
            conversation: The conversation context.
            context: Additional context.

        Returns:
            AgentResult from the selected agent.
        """
        if not resume:
            return AgentResult(
                success=False,
                message="Please upload a resume first before I can help you optimize it. You can upload a PDF or DOCX file.",
                reasoning="No resume uploaded",
                metadata={"agent_type": AgentType.ROUTER.value},
            )

        agent_type = await self._classify_intent(user_message, conversation, context)

        updated_context = self._extract_context(user_message, agent_type, context)

        agent = self._get_agent(agent_type)
        if not agent:
            return await self._handle_general_query(user_message, resume, conversation)

        try:
            result = await agent.process(
                user_message=user_message,
                resume=resume,
                conversation=conversation,
                context=updated_context,
            )
            # Inject agent_type into result metadata for tracking
            result.metadata["agent_type"] = agent_type.value
            return result
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your request.",
                reasoning=f"Agent error: {str(e)}",
                metadata={"agent_type": agent_type.value},
            )

    async def _classify_intent(
        self, user_message: str, conversation: Conversation, context: dict[str, Any]
    ) -> AgentType:
        """
        Classify the user's intent to determine which agent to use.

        Args:
            user_message: The user's message.
            conversation: Conversation context.
            context: Additional context.

        Returns:
            The appropriate AgentType.
        """
        message_lower = user_message.lower()

        for agent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return agent_type

        if context.get("target_company"):
            return AgentType.COMPANY_RESEARCH
        if context.get("job_description"):
            return AgentType.JOB_MATCHING
        if context.get("target_language"):
            return AgentType.TRANSLATION

        return await self._llm_classify_intent(user_message, conversation)

    async def _llm_classify_intent(
        self, user_message: str, conversation: Conversation
    ) -> AgentType:
        """Use LLM to classify intent when patterns don't match."""
        recent_messages = conversation.get_history(limit=3)
        history_context = "\n".join(
            f"{msg.role.value}: {msg.content[:100]}" for msg in recent_messages
        )

        prompt = f"""Classify the user's intent for resume optimization.

Recent conversation:
{history_context}

Current user message: "{user_message}"

Available intents:
1. COMPANY_RESEARCH - User wants to optimize resume for a specific company
2. JOB_MATCHING - User wants to match resume to a job description or analyze fit
3. TRANSLATION - User wants to translate or localize resume for a different market/language

Respond with ONLY one of: COMPANY_RESEARCH, JOB_MATCHING, TRANSLATION, or GENERAL
If the intent is unclear or doesn't fit the above categories, respond with GENERAL."""

        response = await self.llm.ainvoke(prompt)
        intent_str = response.content.strip().upper()

        intent_mapping = {
            "COMPANY_RESEARCH": AgentType.COMPANY_RESEARCH,
            "JOB_MATCHING": AgentType.JOB_MATCHING,
            "TRANSLATION": AgentType.TRANSLATION,
        }

        return intent_mapping.get(intent_str, AgentType.COMPANY_RESEARCH)

    def _extract_context(
        self, user_message: str, agent_type: AgentType, existing_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract relevant context from the user message."""
        context = {**existing_context}

        if agent_type == AgentType.COMPANY_RESEARCH:
            company_patterns = [
                r"(?:for|at|to)\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s*$|\s*\.|\s*,)",
                r"(Google|Amazon|Microsoft|Meta|Apple|Netflix|Spotify|Uber|Airbnb|Tesla|IBM|Oracle|Salesforce)",
            ]
            for pattern in company_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    context["target_company"] = match.group(1).strip()
                    break

        elif agent_type == AgentType.JOB_MATCHING:
            jd_indicators = [
                "job description:",
                "jd:",
                "requirements:",
                "responsibilities:",
            ]
            message_lower = user_message.lower()
            for indicator in jd_indicators:
                if indicator in message_lower:
                    idx = message_lower.find(indicator)
                    context["job_description"] = user_message[idx:]
                    break

            if "job_description" not in context and len(user_message) > 200:
                context["job_description"] = user_message

        elif agent_type == AgentType.TRANSLATION:
            languages = [
                "spanish",
                "french",
                "german",
                "portuguese",
                "italian",
                "japanese",
                "chinese",
                "korean",
                "arabic",
                "hindi",
                "russian",
                "dutch",
            ]
            message_lower = user_message.lower()
            for lang in languages:
                if lang in message_lower:
                    context["target_language"] = lang
                    break

            regions = [
                "Spain",
                "Mexico",
                "France",
                "Germany",
                "Brazil",
                "Japan",
                "China",
                "India",
                "UAE",
                "Canada",
                "Argentina",
                "Italy",
            ]
            for region in regions:
                if region.lower() in message_lower:
                    context["target_region"] = region
                    break

        return context

    async def _handle_general_query(
        self, user_message: str, resume: Resume, conversation: Conversation
    ) -> AgentResult:
        """Handle general queries that don't fit specific agents."""
        prompt = f"""You are a helpful career assistant. The user has uploaded their resume and is asking:

"{user_message}"

Available capabilities:
1. **Company Research & Optimization**: I can research specific companies and optimize your resume to match their culture and values. Example: "Optimize my resume for Google"

2. **Job Description Matching**: I can analyze job descriptions, calculate match scores, identify skill gaps, and optimize your resume for specific positions. Example: "Match my resume to this job description: [paste JD]"

3. **Translation & Localization**: I can translate your resume to different languages and adapt it for specific regional markets. Example: "Translate my resume to Spanish for the Mexican market"

Please help the user understand how to use these features or clarify their request."""

        response = await self.llm.ainvoke(prompt)

        return AgentResult(
            success=True,
            message=response.content,
            reasoning="General query - provided guidance on available features",
            metadata={"agent_type": AgentType.ROUTER.value},
        )

    def get_available_agents(self) -> list[dict[str, str]]:
        """Get information about available agents."""
        return [
            {
                "type": AgentType.COMPANY_RESEARCH.value,
                "name": "Company Research & Optimization",
                "description": "Research companies and optimize your resume to match their culture and values",
                "example": "Optimize my resume for Google",
            },
            {
                "type": AgentType.JOB_MATCHING.value,
                "name": "Job Description Matching",
                "description": "Analyze job descriptions, calculate match scores, and identify skill gaps",
                "example": "Match my resume to this job description: [paste JD]",
            },
            {
                "type": AgentType.TRANSLATION.value,
                "name": "Translation & Localization",
                "description": "Translate and localize your resume for different markets",
                "example": "Translate my resume to Spanish for Mexico",
            },
        ]
