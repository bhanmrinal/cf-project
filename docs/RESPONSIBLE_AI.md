## Responsible Use of AI – Statement

This project was implemented with the help of AI tooling, but final responsibility for the design, code, and documentation rests with the author. The goal of this document is to be transparent about **how** AI was used and **where** human review and judgment were applied.

---

### 1. AI Tools Used

- **ChatGPT & Claude Playground**
  - Used inside browser and occasionally for web UI.
- **Gemini code assistant on VSCode**
  - Used for minor completions and boilerplate (e.g.,pydantic models, api routes, loops, type hints).

No external proprietary code generators (beyond the above) or closed‑source model weights were embedded directly in the repository.

---

### 2. How AI Was Used

AI was used as a **productivity assistant**, not as an autonomous code author. Typical usage patterns:

- **Design & Architecture Ideation**

  - Brainstormed alternative architectures for:
    - Agent routing (rule‑based vs. LLM‑based).
    - Vector store usage (local embeddings vs. Chroma built‑ins).
    - Deployment strategies (Railway, Nixpacks config, image‑size optimizations).
  - Used AI to compare trade‑offs but made final architectural decisions manually.

- **Code Refactoring**

  - Suggested initial scaffolding for:
    - Pydantic models.
    - FastAPI routes and response models.
    - Base agent abstractions and the conversation router.
    - Some business logic in creatind and deploying the pipeline  
  - Helped refactor imports when moving `run.py` and reorganizing the package structure.

- **Debugging & Error Resolution**

  - Used AI to interpret stack traces and logs for:
    - Dependency conflicts (LangChain/Chroma/Firebase versions).
    - `ModuleNotFoundError` issues after refactors.
    - Unicode/encoding issues on Windows.
    - Firebase Firestore query/index warnings.
    - Railway deployment issues (image size, Nixpacks, missing `pip`).
  - AI provided hypotheses and patch suggestions; fixes were validated and adjusted manually.

- **Text Generation for UX**

  - Drafted user‑facing copy such as:
    - Match analysis summaries.
    - Recommendations and strengths in the analysis panel.
    - Descriptions of agents and available capabilities.
  - All user‑visible messages were reviewed and edited for clarity and tone.

---

### 3. What Was _Not_ Delegated to AI

- **Core architectural decisions**

  - Choice of technologies (FastAPI, Groq, ChromaDB, Firebase).
  - Decision to move to pure semantic job matching instead of keyword scoring.
  - LLM routing strategy and safety constraints (e.g., not fabricating skills).

- **Security‑sensitive configuration**

  - Handling of `.env` files and Firebase credentials.
  - Decisions about what is persisted vs. kept in memory.

- **Final code review & integration**
  - All AI‑suggested changes were:
    - Reviewed for correctness and style.
    - Integrated with regard to existing abstractions.
    - Tested locally (and via tests) before being committed.

---

### 4. Risk Mitigation & Validation

To reduce the risk of subtle bugs or hallucinated logic:

- **Manual Testing**

  - End‑to‑end tests of:
    - Resume upload, parsing, and preview.
    - Company optimization flows.
    - Job matching flows with real job descriptions.
    - Translation/localization and export (PDF/DOCX).

- **Automated Tests**

  - `tests/` directory includes focused tests for:
    - Configuration and models.
    - Agent behavior and routing.
    - Export and routing logic.

- **Content Filtering**
  - Export and preview pipelines strip LLM “reasoning” sections (e.g., _Key changes made_), keeping user resumes clean.
  - Agents are instructed not to fabricate experiences or skills, only to rephrase and reorder existing information.

---

### 5. Summary

AI tools were used in this project:

- **For:** brainstorming, refactoring help, debugging support, and draft documentation.
- **Not for:** unreviewed code generation, security‑critical decisions, or unvalidated business logic.

All critical logic and final deliverables were reviewed and validated by the author to ensure they meet the assignment’s expectations and responsible‑use standards.
