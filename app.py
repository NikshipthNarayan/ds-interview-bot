import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import json
import random
from ingest import process_pdf, add_documents_to_vectorstore

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "ds-interview-bot"

# Page configuration
st.set_page_config(
    page_title="DS Interview Prep Bot",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS - Teal/Emerald Theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(13, 148, 136, 0.3);
    }
    .topic-header {
        background: linear-gradient(90deg, #0d9488, #14b8a6);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }
    .quiz-question {
        background: linear-gradient(145deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.1rem;
        color: #064e3b;
    }
    .score-card {
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    .score-number {
        font-size: 3rem;
        font-weight: bold;
    }
    .upload-box {
        background: linear-gradient(145deg, #f0fdfa, #ccfbf1);
        border: 2px dashed #14b8a6;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #0d9488;
    }
    .footer-box {
        background: linear-gradient(135deg, #134e4a 0%, #0d9488 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
    }
    .stat-box {
        background: linear-gradient(145deg, #f0fdfa, #ccfbf1);
        border: 1px solid #5eead4;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0d9488;
    }
    .stat-label {
        color: #115e59;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Quiz questions bank
QUIZ_QUESTIONS = [
    {"question": "What is the main purpose of regularization?", "topic": "ML Fundamentals"},
    {"question": "Explain the bias-variance tradeoff in your own words.", "topic": "ML Fundamentals"},
    {"question": "What is the difference between precision and recall?", "topic": "Model Evaluation"},
    {"question": "When would you use L1 vs L2 regularization?", "topic": "ML Fundamentals"},
    {"question": "What is cross-validation and why is it important?", "topic": "Model Evaluation"},
    {"question": "Explain how gradient descent works.", "topic": "Deep Learning"},
    {"question": "What is overfitting and how do you prevent it?", "topic": "ML Fundamentals"},
    {"question": "What is the difference between bagging and boosting?", "topic": "ML Fundamentals"},
    {"question": "Explain the curse of dimensionality.", "topic": "Data Preprocessing"},
    {"question": "What is PCA and when would you use it?", "topic": "Data Preprocessing"},
    {"question": "How do you handle missing data?", "topic": "Data Preprocessing"},
    {"question": "What is the ROC curve and AUC?", "topic": "Model Evaluation"},
    {"question": "Explain how a Random Forest works.", "topic": "ML Fundamentals"},
    {"question": "What is feature scaling and why is it important?", "topic": "Data Preprocessing"},
    {"question": "What is the difference between classification and regression?", "topic": "ML Fundamentals"},
]

# Initialize session state
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "study_mode" not in st.session_state:
    st.session_state.study_mode = "Learn"
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_total" not in st.session_state:
    st.session_state.quiz_total = 0
if "current_quiz_q" not in st.session_state:
    st.session_state.current_quiz_q = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = 0
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def create_rag_chain(_vectorstore, _llm, mode="Learn"):
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    if mode == "Learn":
        template = """You are an expert Data Science interview coach. Provide a comprehensive, educational answer.

Context: {context}

Question: {question}

Provide a detailed answer that:
1. Explains the concept clearly
2. Includes examples and analogies
3. Mentions real-world applications
4. Adds interview tips where relevant

Answer:"""
    elif mode == "Quick Review":
        template = """You are an expert Data Science interview coach. Provide a concise, bullet-point answer.

Context: {context}

Question: {question}

Provide a brief answer with:
- Key definition (1-2 sentences)
- Main points (3-5 bullets)
- One interview tip

Answer:"""
    else:  # Mock Interview
        template = """You are a tough but fair Data Science interviewer. First provide what a GOOD answer would include, then rate the difficulty.

Context: {context}

Question: {question}

Provide:
1. What interviewers expect in a good answer
2. Common mistakes candidates make
3. Follow-up questions they might ask
4. Difficulty: Easy/Medium/Hard

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def evaluate_answer(user_answer, question, correct_context, llm):
    """Evaluate user's answer and provide feedback."""
    eval_template = """You are an expert Data Science interview evaluator. 

Question: {question}

Reference Answer/Context: {context}

Candidate's Answer: {user_answer}

Evaluate the candidate's answer and provide:
1. A score from 1-10
2. What they got right
3. What they missed or got wrong
4. A brief model answer

Format your response as:
**Score: X/10**

**What you got right:**
[feedback]

**Areas to improve:**
[feedback]

**Model Answer:**
[brief correct answer]"""
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "question": question,
        "context": correct_context,
        "user_answer": user_answer
    })

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 DS Interview Prep Bot</h1>
        <p style="font-size: 1.2rem;">Master Data Science Interviews with AI-Powered Practice</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">INFO 7390 - Advanced Data Science | Northeastern University</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Study Dashboard")
        
        # Main Navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select Section",
            ["💬 Ask Questions", "🧪 Quiz Mode", "📄 Upload Materials"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Study Mode (only for Ask Questions)
        if page == "💬 Ask Questions":
            st.markdown("### 📖 Study Mode")
            mode = st.radio(
                "Select Mode",
                ["Learn", "Quick Review", "Mock Interview"],
                index=["Learn", "Quick Review", "Mock Interview"].index(st.session_state.study_mode),
                label_visibility="collapsed",
                help="Learn: Detailed | Quick Review: Concise | Mock Interview: Simulation"
            )
            st.session_state.study_mode = mode
            
            mode_desc = {
                "Learn": "📚 Detailed explanations with examples",
                "Quick Review": "⚡ Concise bullet-point answers",
                "Mock Interview": "🎤 Interview simulation with tips"
            }
            st.caption(mode_desc[mode])
            
            st.markdown("---")
        
        # Progress Stats
        st.markdown("### 📊 Your Progress")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{st.session_state.total_queries}</div>
                <div class="stat-label">Questions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{st.session_state.uploaded_docs}</div>
                <div class="stat-label">Docs Added</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.quiz_total > 0:
            avg_score = st.session_state.quiz_score / st.session_state.quiz_total
            st.metric("Quiz Average", f"{avg_score:.1f}/10")
        
        progress = min(st.session_state.total_queries / 30, 1.0)
        st.progress(progress)
        st.caption(f"🏆 {int(progress * 100)}% knowledge base explored")
        
        st.markdown("---")
        
        # Bookmarks
        st.markdown("### 🔖 Bookmarked Questions")
        if st.session_state.bookmarks:
            for i, bm in enumerate(st.session_state.bookmarks[-5:]):
                if st.button(f"📌 {bm[:25]}...", key=f"bm_{i}", use_container_width=True):
                    st.session_state.question = bm
            if len(st.session_state.bookmarks) > 5:
                st.caption(f"+ {len(st.session_state.bookmarks) - 5} more")
        else:
            st.caption("Bookmark questions to review later")
        
        st.markdown("---")
        
        # Export
        st.markdown("### 💾 Save Progress")
        if st.session_state.chat_history:
            export_data = {
                "stats": {
                    "total_questions": st.session_state.total_queries,
                    "quiz_score": st.session_state.quiz_score,
                    "quiz_total": st.session_state.quiz_total,
                    "docs_uploaded": st.session_state.uploaded_docs
                },
                "bookmarks": st.session_state.bookmarks,
                "chat_history": st.session_state.chat_history
            }
            st.download_button(
                "📥 Export Study Session",
                data=json.dumps(export_data, indent=2),
                file_name=f"ds_study_session_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.caption("Start practicing to enable export")
    
    # Load components
    try:
        vectorstore = load_vectorstore()
        llm = get_llm()
        rag_chain, retriever = create_rag_chain(vectorstore, llm, st.session_state.study_mode)
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        st.info("Make sure you have run `python ingest.py` first to set up the Pinecone index.")
        return
    
    # =====================================================
    # PAGE 1: ASK QUESTIONS
    # =====================================================
    if page == "💬 Ask Questions":
        st.markdown("### 💡 Suggested Questions")
        
        sample_questions = {
            "ML Fundamentals": ["What is the bias-variance tradeoff?", "Explain supervised vs unsupervised learning", "How does Random Forest work?"],
            "Model Evaluation": ["Explain precision, recall, and F1-score", "What is cross-validation?", "What is the ROC curve?"],
            "Deep Learning": ["Explain gradient descent", "What are activation functions?", "How does backpropagation work?"],
            "Data Preprocessing": ["How do you handle missing data?", "What is feature scaling?", "Explain the curse of dimensionality"]
        }
        
        cols = st.columns(4)
        for idx, (topic, questions) in enumerate(sample_questions.items()):
            with cols[idx]:
                st.markdown(f'<div class="topic-header">{topic}</div>', unsafe_allow_html=True)
                for q in questions:
                    if st.button(q, key=f"btn_{q}", use_container_width=True):
                        st.session_state.question = q
        
        st.markdown("---")
        
        # Input Section
        st.markdown(f"### 🔍 Ask a Question ({st.session_state.study_mode} Mode)")
        
        question = st.text_input(
            "Question:",
            value=st.session_state.question,
            placeholder="Type any data science interview question...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 3])
        with col1:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        with col2:
            bookmark_btn = st.button("🔖 Bookmark", use_container_width=True)
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.question = ""
                st.rerun()
        
        # Handle bookmark
        if bookmark_btn and question:
            if question not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(question)
                st.toast("✅ Question bookmarked!")
            else:
                st.toast("Already bookmarked!")
        
        # Process question
        if ask_button and question:
            st.session_state.total_queries += 1
            
            with st.spinner(f"🤔 Generating {st.session_state.study_mode} response..."):
                try:
                    answer = rag_chain.invoke(question)
                    retrieved_docs = retriever.invoke(question)
                    
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "answer": answer,
                        "mode": st.session_state.study_mode
                    })
                    
                    # Display answer
                    st.markdown("### 📝 Answer")
                    st.markdown(answer)
                    
                    # Sources
                    with st.expander("📚 Knowledge Base Sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.info(doc.page_content[:400] + "...")
                    
                    # Tip based on mode
                    tips = {
                        "Learn": "💡 **Study Tip:** Try explaining this concept to someone else - teaching reinforces learning!",
                        "Quick Review": "⚡ **Review Tip:** Create flashcards for quick revision before interviews!",
                        "Mock Interview": "🎤 **Interview Tip:** Practice answering out loud within 2 minutes!"
                    }
                    st.success(tips[st.session_state.study_mode])
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif ask_button and not question:
            st.warning("⚠️ Please enter a question first!")
    
    # =====================================================
    # PAGE 2: QUIZ MODE
    # =====================================================
    elif page == "🧪 Quiz Mode":
        st.markdown("### 🧪 Test Your Knowledge")
        st.markdown("The AI will ask you a question. Type your answer and get instant feedback with a score!")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{st.session_state.quiz_score:.0f}</div>
                <div>Total Points</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{st.session_state.quiz_total} questions attempted</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.quiz_total > 0:
                avg = st.session_state.quiz_score / st.session_state.quiz_total
                st.markdown(f"**Average: {avg:.1f}/10**")
                
                if avg >= 8:
                    st.success("🌟 Excellent!")
                elif avg >= 6:
                    st.info("👍 Good progress!")
                else:
                    st.warning("📚 Keep practicing!")
        
        with col1:
            # Generate new question button
            if st.button("🎲 Get New Question", type="primary", use_container_width=True):
                st.session_state.current_quiz_q = random.choice(QUIZ_QUESTIONS)
            
            # Display current question
            if st.session_state.current_quiz_q:
                q = st.session_state.current_quiz_q
                st.markdown(f"""
                <div class="quiz-question">
                    <strong>📋 Question ({q['topic']}):</strong><br><br>
                    <span style="font-size: 1.2rem;">{q['question']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # User answer input
                user_answer = st.text_area(
                    "Your Answer:",
                    height=150,
                    placeholder="Type your answer here... Be as detailed as you would in a real interview.",
                    key="quiz_answer"
                )
                
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    submit_btn = st.button("✅ Submit Answer", type="primary", use_container_width=True)
                with col_b:
                    skip_btn = st.button("⏭️ Skip Question", use_container_width=True)
                
                if skip_btn:
                    st.session_state.current_quiz_q = random.choice(QUIZ_QUESTIONS)
                    st.rerun()
                
                if submit_btn:
                    if user_answer.strip():
                        with st.spinner("🔍 Evaluating your answer..."):
                            # Get context from knowledge base
                            context_docs = retriever.invoke(q['question'])
                            context = "\n".join([d.page_content for d in context_docs])
                            
                            # Evaluate
                            feedback = evaluate_answer(user_answer, q['question'], context, llm)
                            
                            # Extract score
                            try:
                                score_line = [l for l in feedback.split('\n') if 'Score:' in l][0]
                                score = float(score_line.split('/')[0].split(':')[-1].strip())
                                st.session_state.quiz_score += score
                                st.session_state.quiz_total += 1
                            except:
                                score = 5
                                st.session_state.quiz_score += score
                                st.session_state.quiz_total += 1
                            
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "question": q['question'],
                                "user_answer": user_answer,
                                "feedback": feedback,
                                "score": score,
                                "mode": "quiz"
                            })
                            
                            # Display feedback
                            st.markdown("### 📊 Your Feedback")
                            
                            if score >= 8:
                                st.success(f"🌟 Excellent! Score: {score}/10")
                                st.balloons()
                            elif score >= 6:
                                st.info(f"👍 Good job! Score: {score}/10")
                            else:
                                st.warning(f"📚 Keep practicing! Score: {score}/10")
                            
                            st.markdown(feedback)
                            
                            st.markdown("---")
                            st.info("💡 Click 'Get New Question' to continue practicing!")
                    else:
                        st.warning("Please write an answer first!")
            else:
                st.info("👆 Click **'Get New Question'** to start the quiz!")
                
                st.markdown("---")
                st.markdown("#### How Quiz Mode Works:")
                st.markdown("""
                1. Click **'Get New Question'** to receive a random interview question
                2. Type your answer as if you're in a real interview
                3. Click **'Submit Answer'** to get AI feedback
                4. Receive a score (1-10) with detailed feedback
                5. Track your progress in the sidebar!
                """)
    
    # =====================================================
    # PAGE 3: UPLOAD MATERIALS
    # =====================================================
    elif page == "📄 Upload Materials":
        st.markdown("### 📄 Expand Your Knowledge Base")
        st.markdown("Add your own study materials! The bot will use YOUR content to answer questions.")
        
        tab1, tab2 = st.tabs(["📤 Upload PDF", "📝 Paste Text"])
        
        with tab1:
            st.markdown("""
            <div class="upload-box">
                <h3>📤 Upload PDF Files</h3>
                <p>Add textbooks, notes, or any DS/ML content to enhance the knowledge base</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.success(f"**{len(uploaded_files)} file(s) selected**")
                
                for f in uploaded_files:
                    st.markdown(f"- 📄 {f.name}")
                
                if st.button("📥 Add to Knowledge Base", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    total_chunks = 0
                    for i, file in enumerate(uploaded_files):
                        status.text(f"Processing {file.name}...")
                        
                        try:
                            text = process_pdf(file)
                            chunks = add_documents_to_vectorstore([text], source_name=file.name)
                            total_chunks += chunks
                            st.session_state.uploaded_docs += 1
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Successfully added {total_chunks} chunks from {len(uploaded_files)} file(s)!")
                    st.info("💡 Go to 'Ask Questions' and ask about your uploaded content!")
                    
                    load_vectorstore.clear()
        
        with tab2:
            st.markdown("#### Paste Your Notes")
            st.markdown("Directly paste text content to add to the knowledge base.")
            
            text_input = st.text_area(
                "Paste your notes here:",
                height=250,
                placeholder="Paste any text content you want to add...\n\nFor example:\n- Class notes\n- Article summaries\n- Key concepts you want to remember"
            )
            
            source_name = st.text_input("Source name (optional):", placeholder="e.g., ML_lecture_notes")
            
            if st.button("➕ Add Text to Knowledge Base", type="primary", use_container_width=True):
                if text_input.strip():
                    with st.spinner("Adding to knowledge base..."):
                        try:
                            name = source_name if source_name else "pasted_notes"
                            chunks = add_documents_to_vectorstore([text_input], source_name=name)
                            st.session_state.uploaded_docs += 1
                            st.success(f"✅ Added {chunks} chunks to knowledge base!")
                            st.info("💡 Go to 'Ask Questions' and ask about your added content!")
                            load_vectorstore.clear()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please paste some text first!")
        
        st.markdown("---")
        st.markdown("### 📊 Upload Statistics")
        st.markdown(f"**Total documents added:** {st.session_state.uploaded_docs}")
        
        if st.session_state.uploaded_docs > 0:
            st.success("Your knowledge base has been expanded! Try asking questions about your uploaded content.")
    
    # Footer
    st.markdown("""
    <div class="footer-box">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🧠 DS Interview Prep Bot</p>
        <p style="opacity: 0.9;">Built with OpenAI GPT-4o-mini • Pinecone • LangChain • Streamlit</p>
        <p style="opacity: 0.7; font-size: 0.85rem;">Nikshipth Narayan | INFO 7390 - Northeastern University</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()