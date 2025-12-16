# Careerflow Resume Optimization System

A conversational AI system for resume optimization using specialized agents. Built with FastAPI, LangChain, and open-source LLMs (Llama 3.3 via Groq).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)

## 🚀 Features

### Specialized Agents

1. **Company Research & Optimization Agent** 🏢

   - Researches target companies using web search (DuckDuckGo)
   - Analyzes company culture, values, and hiring patterns
   - Optimizes resume content to match company preferences
   - Adjusts language and emphasis based on company research

2. **Job Description Matching Agent** 🎯

   - Analyzes job descriptions to extract requirements
   - Calculates match scores (required skills, preferred skills, keywords)
   - Identifies skill gaps and missing qualifications
   - Restructures resume to highlight relevant experience
   - Provides ATS optimization recommendations

3. **Translation & Localization Agent** 🌍
   - Translates resumes to 12+ languages
   - Adapts content for regional cultural contexts
   - Applies local resume formatting conventions
   - Supports markets: Spain, Mexico, France, Germany, Japan, China, India, UAE, and more

### Core Capabilities

- **Intelligent Conversation Router**: Automatically routes requests to the appropriate agent
- **Resume Parsing**: Supports PDF and DOCX formats with section extraction
- **Version Control**: Track all resume changes with undo/redo functionality
- **Vector Search**: Semantic search using ChromaDB for context retrieval
- **Real-time Chat Interface**: Modern, responsive web UI

## 🏗️ Architecture

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
│         (Intent Classification & Agent Selection)                │
└─────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Company Research│  │  Job Matching   │  │  Translation    │
│     Agent       │  │     Agent       │  │     Agent       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Groq     │  │ HuggingFace │  │   Ollama    │             │
│  │ (Llama 3.3) │  │  (Optional) │  │  (Local)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Firebase     │  │    ChromaDB     │  │  File Storage   │
│  (Conversations)│  │ (Vector Store)  │  │   (Uploads)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 📁 Project Structure

```
careerflow-project/
├── backend/
│   └── app/
│       ├── agents/                 # Specialized AI agents
│       │   ├── base.py            # Base agent class
│       │   ├── company_research.py # Company research agent
│       │   ├── job_matching.py    # Job matching agent
│       │   ├── translation.py     # Translation agent
│       │   └── router.py          # Conversation router
│       ├── api/
│       │   └── routes/            # API endpoints
│       │       ├── chat.py        # Chat endpoints
│       │       ├── resume.py      # Resume endpoints
│       │       └── conversation.py # Conversation endpoints
│       ├── core/
│       │   ├── config.py          # Application configuration
│       │   └── llm.py             # LLM factory
│       ├── models/                # Pydantic models
│       │   ├── resume.py          # Resume models
│       │   ├── conversation.py    # Conversation models
│       │   └── chat.py            # Chat request/response models
│       ├── services/              # Business logic services
│       │   ├── resume_parser.py   # PDF/DOCX parsing
│       │   ├── firebase_service.py # Firebase operations
│       │   └── vector_store.py    # ChromaDB operations
│       └── main.py                # FastAPI application
├── frontend/
│   ├── index.html                 # Main HTML
│   ├── styles.css                 # Styles
│   └── app.js                     # Frontend JavaScript
├── requirements.txt               # Python dependencies
├── run.py                         # Entry point
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Groq API key (free at https://console.groq.com/)
- Optional: Firebase project for persistent storage

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd careerflow-project
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   ```bash
   # Create .env file
   copy .env.example .env   # Windows
   cp .env.example .env     # Linux/Mac
   ```

   Edit `.env` with your settings:

   ```env
   # Required: Groq API Key (free at https://console.groq.com/)
   GROQ_API_KEY=your_groq_api_key_here

   # LLM Configuration
   LLM_PROVIDER=groq
   GROQ_MODEL=llama-3.3-70b-versatile

   # Optional: Firebase (for persistent storage)
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_PRIVATE_KEY=your_private_key
   FIREBASE_CLIENT_EMAIL=your_client_email
   ```

5. **Run the application**

   ```bash
   python run.py
   ```

6. **Open in browser**
   ```
   http://localhost:8000
   ```

## 🔧 Configuration

### LLM Providers

The system supports multiple LLM providers:

| Provider           | Model                            | Notes                            |
| ------------------ | -------------------------------- | -------------------------------- |
| **Groq** (Default) | llama-3.3-70b-versatile          | Fast, free tier available        |
| Groq               | mixtral-8x7b-32768               | Alternative model                |
| HuggingFace        | meta-llama/Llama-3.2-3B-Instruct | Requires HF API key              |
| Ollama             | llama3.2                         | Local, requires Ollama installed |

### Environment Variables

| Variable                   | Description           | Default                   |
| -------------------------- | --------------------- | ------------------------- |
| `LLM_PROVIDER`             | LLM provider to use   | `groq`                    |
| `GROQ_API_KEY`             | Groq API key          | Required for Groq         |
| `GROQ_MODEL`               | Groq model name       | `llama-3.3-70b-versatile` |
| `HUGGINGFACE_API_KEY`      | HuggingFace API key   | Optional                  |
| `OLLAMA_BASE_URL`          | Ollama server URL     | `http://localhost:11434`  |
| `FIREBASE_*`               | Firebase credentials  | Optional                  |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./chroma_db`             |

## 📖 API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Chat

- `POST /api/chat/message` - Send a chat message
- `GET /api/chat/agents` - List available agents

#### Resume

- `POST /api/resume/upload` - Upload a resume (PDF/DOCX)
- `GET /api/resume/{id}` - Get resume details
- `GET /api/resume/{id}/versions` - Get version history
- `POST /api/resume/{id}/revert/{version}` - Revert to version

#### Conversation

- `GET /api/conversation/{id}` - Get conversation history
- `DELETE /api/conversation/{id}` - Delete conversation

## 💬 Usage Examples

### Optimize for a Company

```
User: Optimize my resume for Google

System: I've researched Google and optimized your resume to match their culture
and values.

Key changes:
- Emphasized collaborative project experience
- Added metrics to demonstrate impact
- Aligned language with Google's innovation focus
```

### Match to Job Description

```
User: Match my resume to this job description: [paste JD]

System: 📊 Match Analysis Complete
Overall Match Score: 72%
- Required Skills: 80%
- Keywords: 65%

Skill Gaps Identified:
• Kubernetes experience
• AWS certifications

I've optimized your resume to better highlight relevant experience.
```

### Translate Resume

```
User: Translate my resume to Spanish for the Mexican market

System: 🌍 Translation Complete
Target Language: Spanish
Target Region: Mexico

Regional Conventions Applied:
• Photo: Often expected
• Format: Similar to US but more personal info

Your resume has been translated and localized.
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

## 🔒 Security Considerations

- API keys are stored in environment variables, never committed to code
- File uploads are validated for type and size
- Firebase credentials use service accounts with minimal permissions
- CORS is configured (adjust for production)

## 🚀 Deployment on Railway

Deploy your own instance in minutes using Railway:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

### Steps

1. **Push to GitHub** - Make sure your code is in a GitHub repository

2. **Sign up at Railway** - Go to [railway.app](https://railway.app) and sign up with GitHub

3. **Create New Project**

   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Add Environment Variables**

   - Go to your service → Variables tab
   - Add the following:

   | Variable       | Required | Value                                                    |
   | -------------- | -------- | -------------------------------------------------------- |
   | `GROQ_API_KEY` | ✅       | Get free at [console.groq.com](https://console.groq.com) |
   | `APP_ENV`      | ❌       | `production`                                             |

5. **Deploy** - Railway auto-detects Python and deploys automatically

6. **Get Your URL** - Once deployed, Railway provides a public URL like `your-app.up.railway.app`

### Railway Configuration

The project includes `railway.json` and `Procfile` for automatic configuration:

```json
// railway.json
{
  "build": { "builder": "NIXPACKS" },
  "deploy": {
    "startCommand": "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"
  }
}
```

## 🚧 Future Improvements

- [ ] Add authentication/user management
- [ ] Implement WebSocket for real-time updates
- [ ] Add more language support
- [ ] Integrate with job boards for automatic matching
- [ ] Add resume templates and formatting options
- [ ] Implement caching for company research

