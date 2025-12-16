"""
Agents module for specialized resume optimization agents.

Each agent is responsible for a specific type of resume optimization:
- CompanyResearchAgent: Research companies and optimize resumes for them
- JobMatchingAgent: Match resumes to job descriptions
- TranslationAgent: Translate and localize resumes
"""

from backend.app.agents.base import BaseAgent, AgentResult
from backend.app.agents.company_research import CompanyResearchAgent
from backend.app.agents.job_matching import JobMatchingAgent
from backend.app.agents.translation import TranslationAgent
from backend.app.agents.router import ConversationRouter

__all__ = [
    "BaseAgent",
    "AgentResult",
    "CompanyResearchAgent",
    "JobMatchingAgent",
    "TranslationAgent",
    "ConversationRouter",
]
