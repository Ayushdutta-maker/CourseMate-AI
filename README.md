# CourseMate AI — Intelligent Academic Assistant & Study Companion

A master's-level (MCA AI & ML) intelligent academic learning and revision platform. **CourseMate AI** transforms raw academic curricula, lecture notes, and research documents into structured study manuscripts, interactive multiple-choice quizzes, career roadmaps, and document-grounded question-answering pipelines.

Designed with an **Editorial University & Academic SaaS** aesthetic inspired by prestigious academic institutions — blending warm ivory (`#F8F7F3`), deep academic navy (`#17253C`), and antique gold (`#C5A66A`) with classical serif typography (*Playfair Display*) and clean modern interfaces (*Inter*).

---

## Key Modules & Features

### 1. Editorial Public Landing Portal (`/`)
* **Hero Architecture**: Full-width academic library photograph with an atmospheric dark/warm overlay, elegant typography, and direct action CTAs (*"Explore AI Tools"* & *"Start Learning"*).
* **Curricular Problem Solving**: Concise introduction explaining how CourseMate synthesizes complex academic research into digestible learning modules.
* **Feature Grid**: Showcase of the 6 core modules with direct links and icon cues.
* **How It Works**: Visual five-step pipeline explaining the text extraction, TF-IDF vector retrieval, neural synthesis, and revision workflow.
* **Course Discovery**: University-style course cards filterable by discipline (*AI & Machine Learning*, *Data Science*, *Python Programming*, *CS Fundamentals*, *Cybersecurity*, *Mathematics & Statistics*).

### 2. Main Academic Workspace Dashboard (`/dashboard`)
* **Real Historical Telemetry**: Dynamic summary counters derived directly from persisted database records (Total Sessions, Notes Generated, Quizzes Completed, Career Roadmaps).
* **Quick Launch Modules**: One-click access to AI Assistant, Study Notes, Quiz Lab, and Pathfinder.
* **Academic Footprint Log**: Chronological audit trail with modal inspection to view full saved manuscripts, quiz keys, and conversations.
* **Active Corpus Status**: Live badge indicating whether a custom lecture document is currently active for RAG operations.

### 3. AI Academic Assistant Tutor (`/chat`)
* **Scholarly Dialog**: Built for mathematical derivations, algorithm step-throughs, and conceptual explanations.
* **Suggested Academic Prompts**: Quick-start pills for common revision inquiries (e.g., *Supervised vs Unsupervised ML*, *Transformer Attention*, *Bayes' Theorem Proof*).
* **Document Grounding (RAG) Toggle**: Switch between global scholarly knowledge and strict document-grounded context.
* **Formatted Output**: Full Markdown rendering supporting code blocks, equations, lists, and tables.

### 4. Study Notes Generator (`/notes`)
* **Multi-Input Synthesis**: Accept a topic title or upload a lecture document (`.pdf` or `.txt`) with drag-and-drop support.
* **Academic Level & Synthesis Mode**: Configurable for Undergraduate, Postgraduate (MCA/M.Tech), or Doctoral levels in Comprehensive, Deep Dive, or Exam Viva modes.
* **Curricular Manuscript Output**: Structured with Introduction, Definitions, Principles, Examples, Revision Points, and Exam Questions.
* **Export Utilities**: One-click Copy to Markdown, Save as `.md` file, and Print / PDF export.
* **Integrated Document RAG Q&A**: Follow-up question box directly grounded in the active manuscript using TF-IDF vector similarity.

### 5. Interactive Quiz Practice Lab (`/quiz`)
* **Configurable Assessment**: Select topic and question volume (5, 8, or 10 questions).
* **Step-by-Step Question Taker**: Progress bar, option selection with active states, and ability to navigate between questions.
* **Instant Evaluation Engine**: Automatically computes raw score, percentage, and assigns an academic mastery grade.
* **Answer Review Accordion**: Question-by-question breakdown showing selected answers, correct answer keys, and pedagogical explanations.
* **Retake / New Topic Options**: Retake the same quiz or configure a new subject assessment.

### 6. Career Pathfinder & Course Matcher (`/recommendations`)
* **Career Trajectory Synthesizer**: Formulates phased milestones, required competencies, capstone project suggestions, and a 30-day action plan based on user goals and time availability.
* **Vector Course Matcher**: Uses scikit-learn's `TfidfVectorizer` and `cosine_similarity` to calculate match percentages against the academic course catalog.

### 7. Learning History Vault (`/history`)
* **Persistent Records**: All chat sessions, study notes, quiz evaluations, and roadmaps are stored in the database.
* **Discipline Filtering & Search**: Instant filtering by module type or keyword search.
* **Full Content Inspection**: Open any historical record in a modal viewer with full formatting preserved.

---

## Technical Architecture

```
[ User Browser / Desktop / Mobile ]
                 │
                 ▼
[ Flask Application (Python 3.11) — app.py ]
  ├── Routes: /, /dashboard, /chat, /notes, /quiz, /recommendations, /history
  ├── REST APIs: /api/chat, /api/generate-notes, /api/rag-chat, /api/generate-quiz, /api/recommend, /api/history
  │
  ├── [ Database Layer ]
  │     ├── SQLAlchemy ORM (History Model)
  │     └── Automatic Fallback: Configured MySQL ➔ SQLite (instance/database.db)
  │
  └── [ Academic AI & RAG Engine — utils.py ]
        ├── NVIDIA NIM API (Llama / Gemma models)
        ├── Google Gemini API (Configured fallback)
        ├── Scikit-learn TF-IDF Vectorizer & Cosine Similarity
        ├── PyPDF2 & Unicode Text Extraction
        └── Local Academic Synthesis Engine (100% offline fallback reliability)
```

---

## Setup & Running Locally

### 1. Prerequisites
- Python 3.10+ or Python 3.11+
- Virtual environment (recommended) or system Python

### 2. Environment Configuration
Inspect or create `.env` in the project root:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=coursemate_academic_ai_secure_2026
# Optional MySQL connection string (falls back cleanly to SQLite if not provided)
# DATABASE_URL=mysql+pymysql://root:password@localhost/coursemate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## Verification & Testing Guide

To verify end-to-end functionality across all routes and API endpoints:

1. **Public Portal**: Visit `http://127.0.0.1:5000/` — check Hero section, smooth anchor navigation, category filtering under Course Recommendations.
2. **Dashboard**: Navigate to `http://127.0.0.1:5000/dashboard` — verify total session count (28+ records loaded from database), module cards, and recent activity table. Click "View" to open the modal.
3. **AI Assistant**: Navigate to `http://127.0.0.1:5000/chat` — click a suggestion pill (e.g. *Supervised vs Unsupervised ML*), verify Markdown response.
4. **Study Notes**: Navigate to `http://127.0.0.1:5000/notes` — enter a topic (e.g. *Operating System Deadlocks*) and click *Synthesize Study Manuscript*. Test copy, download, and the RAG follow-up Q&A box.
5. **Quiz Lab**: Navigate to `http://127.0.0.1:5000/quiz` — enter a subject (e.g. *Machine Learning*), answer questions, click *Submit Assessment*, and inspect score breakdown and explanations.
6. **Career Pathfinder**: Navigate to `http://127.0.0.1:5000/recommendations` — enter goals and generate roadmap. Switch to the *Course Vector Matcher* tab to test TF-IDF cosine similarity.
7. **Learning History**: Navigate to `http://127.0.0.1:5000/history` — filter by module type and test searching by topic.
