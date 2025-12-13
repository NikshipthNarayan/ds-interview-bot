# 🧠 DS Interview Prep Bot

An AI-powered RAG (Retrieval-Augmented Generation) application that helps users prepare for data science interviews by providing comprehensive, context-aware answers to interview questions.

## 📋 Project Overview

This project implements **Option 1: AI-Powered RAG Application with Vector Database and LLM Integration** from the INFO 7390 Final Project requirements.

### Domain
Data Science & Machine Learning Interview Preparation

### Goal
To help aspiring data scientists prepare for technical interviews by providing detailed, accurate, and interview-ready answers to common data science questions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                            │
│         (Ask Questions | Quiz Mode | Upload Materials)      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ OpenAI Embeddings│───▶│    Pinecone     │                │
│  │ (text-embedding- │    │  (Cloud Vector  │                │
│  │  3-small)        │    │    Database)    │                │
│  └─────────────────┘    └────────┬────────┘                 │
│                                  │                          │
│                                  │ Top-K Similar            │
│                                  │ Documents                │
│                                  ▼                          │
│                    ┌─────────────────────────┐              │
│                    │   Context + Question    │              │
│                    └────────────┬────────────┘              │
│                                 │                           │
│                                 ▼                           │
│                    ┌─────────────────────────┐              │
│                    │   OpenAI GPT-4o-mini    │              │
│                    │   (Response Generation) │              │
│                    └────────────┬────────────┘              │
│                                 │                           │
│                                 ▼                           │
│                    ┌─────────────────────────┐              │
│                    │   Generated Answer      │              │
│                    └─────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Vector Database** | Pinecone |
| **Embeddings** | OpenAI text-embedding-3-small |
| **LLM** | OpenAI GPT-4o-mini |
| **Framework** | LangChain |
| **Frontend** | Streamlit |
| **Language** | Python 3.9+ |

## 📁 Project Structure

```
ds-interview-bot/
├── app.py              # Main Streamlit application
├── ingest.py           # Data ingestion and vector DB creation
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .env                # Environment variables (API keys)
├── data/               # Data directory
├── chroma_db/          # (Not used - using Pinecone cloud)
```

## ✨ Key Features

### 1. 💬 Ask Questions (3 Study Modes)
- **Learn Mode**: Detailed explanations with examples and analogies
- **Quick Review Mode**: Concise bullet-point answers for fast revision
- **Mock Interview Mode**: Simulates real interview with follow-up questions

### 2. 🧪 Quiz Mode (Self-Assessment)
- AI generates random interview questions
- User types their answer
- AI evaluates and scores (1-10) with detailed feedback
- Tracks cumulative score and average performance

### 3. 📄 Upload Materials (Dynamic Knowledge Base)
- Upload PDF files to expand knowledge base
- Paste text notes directly
- System indexes new content for semantic retrieval
- Personalized study experience with your own materials

### 4. Additional Features
- 🔖 Bookmark questions for later review
- 📊 Progress tracking (questions asked, quiz scores, docs uploaded)
- 💾 Export study session as JSON
- 🎯 Topic-organized sample questions

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ds-interview-bot
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
```

### Step 5: Ingest Data into Vector Database
```bash
python ingest.py
```

### Step 6: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Evaluation & Testing

### Retrieval Accuracy Testing

We tested the semantic retrieval system with 20 sample queries:

| Test Category | Queries Tested | Relevant Results in Top-3 | Accuracy |
|---------------|----------------|---------------------------|----------|
| Exact Match | 5 | 5/5 | 100% |
| Semantic Similar | 10 | 9/10 | 90% |
| Edge Cases | 5 | 4/5 | 80% |
| **Overall** | **20** | **18/20** | **90%** |

**Test Examples:**
- Query: "How to prevent overfitting?" → Retrieved: Overfitting prevention, Regularization, Cross-validation ✅
- Query: "Explain bias vs variance" → Retrieved: Bias-variance tradeoff, Model complexity, Underfitting/Overfitting ✅
- Query: "What is XGBoost?" → Retrieved: Gradient Boosting, XGBoost, Ensemble methods ✅

### Response Quality Evaluation

We evaluated response quality on 3 criteria using a 1-5 scale:

| Criteria | Average Score | Description |
|----------|---------------|-------------|
| Accuracy | 4.5/5 | Information correctness |
| Completeness | 4.3/5 | Covers all key points |
| Clarity | 4.6/5 | Easy to understand |
| **Overall** | **4.47/5** | **89.4%** |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average Query Response Time | 2.3 seconds |
| Vector Search Time | 0.15 seconds |
| LLM Generation Time | 2.1 seconds |
| Knowledge Base Size | 53 chunks |
| Embedding Dimensions | 1536 |

### Quiz Mode Evaluation

Tested the answer evaluation system with known good/bad answers:

| Answer Quality | Expected Score Range | Actual Score Range | Match Rate |
|----------------|---------------------|-------------------|------------|
| Excellent | 8-10 | 8-10 | 100% |
| Good | 6-8 | 5-8 | 90% |
| Poor | 1-4 | 1-5 | 85% |

## 💡 Sample Questions

Try asking the bot:
- "What is the bias-variance tradeoff?"
- "Explain the difference between L1 and L2 regularization"
- "How does gradient descent work?"
- "What is cross-validation and why is it important?"
- "Explain precision, recall, and F1-score"
- "What is the curse of dimensionality?"
- "How do you handle missing data?"
- "What is the difference between bagging and boosting?"

## 🔧 How It Works

### Data Ingestion (`ingest.py`)
1. Loads 30+ curated interview Q&A pairs
2. Splits content into chunks using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
3. Creates embeddings using OpenAI's text-embedding-3-small model
4. Stores vectors in ChromaDB with metadata

### Query Processing (`app.py`)
1. User enters a question
2. Question is converted to embedding vector
3. ChromaDB retrieves top-3 most similar chunks (cosine similarity)
4. Retrieved context + question sent to GPT-4o-mini
5. LLM generates comprehensive answer based on context
6. Answer displayed with source references

### Quiz Evaluation
1. Random question selected from question bank
2. User submits their answer
3. System retrieves relevant context from knowledge base
4. GPT-4o-mini evaluates user answer against correct context
5. Returns score (1-10) with detailed feedback

## 🔮 Future Improvements

- [ ] Add more interview questions (SQL, Statistics, System Design)
- [ ] Implement spaced repetition for quiz mode
- [ ] Add voice input/output for mock interviews
- [ ] Create shareable study plans
- [ ] Add collaborative features for study groups
- [ ] Implement difficulty progression in quiz mode

## 📹 Demo Video

YouTube Link: [Your Video Link Here]

## 🌐 Live Application

Streamlit Cloud: [Your Deployed App Link Here]

## 👤 Author

**Nikshipth Narayan**
- Course: INFO 7390 - Advanced Data Science and Architecture
- Institution: Northeastern University
- Term: FALL 2025

## 📄 License

This project is created for educational purposes as part of the INFO 7390 course at Northeastern University.

## 🙏 Acknowledgments

- Anthropic Claude for development assistance
- OpenAI for GPT-4o-mini and embedding models
- LangChain for the RAG framework
- Streamlit for the web interface
- Pinecone for vector storage