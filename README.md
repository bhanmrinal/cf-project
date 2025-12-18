# Resume Optimization System

A conversational AI system for resume optimization using specialized agents. Built with FastAPI, LangChain, and Groq (Llama 3.3 70B).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)

## Features

### Specialized Agents

1. **Company Research & Optimization Agent**

   - Researches target companies using web search (DuckDuckGo)
   - Analyzes company culture, values, and hiring patterns
   - Optimizes resume content to match company preferences
   - Adjusts language and emphasis based on company research

2. **Job Description Matching Agent**

   - Analyzes job descriptions to extract requirements
   - Calculates match scores (required skills, preferred skills, keywords)
   - Identifies skill gaps and missing qualifications
   - Restructures resume to highlight relevant experience
   - Provides ATS optimization recommendations

3. **Translation & Localization Agent**
   - Translates resumes to 12+ languages
   - Adapts content for regional cultural contexts
   - Applies local resume formatting conventions
   - Supports markets: Spain, Mexico, France, Germany, Japan, China, India, UAE, and more

### Core Capabilities

- **LLM-Based Intent Routing**: Intelligently routes requests to the appropriate agent using LLM classification
- **Resume Parsing**: Supports PDF and DOCX formats with section extraction
- **Version Control**: Track all resume changes with undo/redo functionality
- **Vector Search**: Semantic search using ChromaDB for context retrieval
- **Real-time Chat Interface**: Modern, responsive web UI
- **Firebase Integration**: Persistent storage for conversations and resumes

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/CSS/JS)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Chat API   │  │ Resume API  │  │ Conv. API   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Router                           │
│            (LLM-Based Intent Classification)                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Company Research│  │  Job Matching   │  │  Translation    │
│     Agent       │  │     Agent       │  │     Agent       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Groq LLM (Llama 3.3 70B)                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Firebase     │  │    ChromaDB     │  │  File Storage   │
│  (Conversations)│  │ (Vector Store)  │  │   (Uploads)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Project Structure

```
cf-project/
├── backend/
│   ├── app/
│   │   ├── agents/           # Specialized AI agents
│   │   │   ├── base.py       # Base agent class
│   │   │   ├── company_research.py
│   │   │   ├── job_matching.py
│   │   │   ├── translation.py
│   │   │   └── router.py     # LLM-based conversation router
│   │   ├── api/
│   │   │   └── routes/       # API endpoints
│   │   │       ├── chat.py
│   │   │       ├── conversation.py
│   │   │       └── resume.py
│   │   ├── core/
│   │   │   ├── config.py     # Application settings
│   │   │   └── llm.py        # Groq LLM factory
│   │   ├── models/           # Pydantic models
│   │   ├── services/         # Business logic services
│   │   │   ├── firebase_service.py
│   │   │   ├── resume_parser.py
│   │   │   └── vector_store.py
│   │   └── main.py           # FastAPI application
│   └── run.py                # Application entry point
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/                    # Test suite
│   ├── test_agents.py        # Agent integration tests
│   ├── test_routing.py       # LLM routing tests
│   ├── test_export.py        # Export functionality tests
│   ├── test_config.py        # Configuration tests
│   └── test_models.py        # Model tests
├── docs/                     # Documentation
├── output/                   # Generated exports (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (free at https://console.groq.com/)
- Firebase project (optional, falls back to in-memory storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cf-project.git
cd cf-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional: Firebase (for persistent storage)
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_PRIVATE_KEY_ID=your_key_id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your_client_id
FIREBASE_CLIENT_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/...
```

### Running the Application

```bash
cd backend
python run.py
```

The application will be available at:
- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Endpoints

### Resume Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/resume/upload` | Upload a resume (PDF/DOCX) |
| GET | `/api/resume/{id}` | Get resume by ID |
| POST | `/api/resume/{id}/export` | Export resume (PDF/DOCX) |
| POST | `/api/resume/{id}/analyze` | Analyze resume with LLM |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/message` | Send a message to the AI |
| GET | `/api/chat/agents` | List available agents |

### Conversation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversation/{id}` | Get conversation history |
| DELETE | `/api/conversation/{id}` | Delete a conversation |

## Testing

```bash
# Run unit tests
pytest tests/test_config.py tests/test_models.py -v

# Run integration tests (requires running server)
python tests/test_agents.py
python tests/test_routing.py
```

## Deployment

### Railway

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

The project includes `railway.json` and `Procfile` for automatic deployment.

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Database**: Firebase Firestore
- **Frontend**: Vanilla HTML/CSS/JS

## License

MIT License - see [LICENSE](LICENSE) for details.
