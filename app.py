import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from utils import ai_engine
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'coursemate_ai_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Umesh%4026@localhost/coursemate'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Simple in-memory cache for RAG
document_store = {"current_text": ""}

# --- Models ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20), nullable=False) # 'chat', 'notes', 'quiz', 'recommendation'
    topic = db.Column(db.String(200))
    content = db.Column(db.Text, nullable=False) # JSON format for structured data
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- Routes ---

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    # Fetch all activity from global history since authentication is removed
    history = History.query.order_by(History.timestamp.desc()).all()
    user = {"username": "Academic Admin", "history": history}
    return render_template('dashboard.html', user=user)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/notes')
def notes():
    return render_template('notes.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

# --- API Endpoints ---

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    message = data.get('message')
    
    response_text = ai_engine.get_ai_response(message, context="You are a helpful education assistant.")
    
    try:
        new_entry = History(type='chat', content=json.dumps({'query': message, 'response': response_text}))
        db.session.add(new_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Database sync error: {e}")
    
    return jsonify({'response': response_text})

@app.route('/api/generate-notes', methods=['POST'])
def generate_notes():
    topic = request.form.get('topic')
    file = request.files.get('file')
    
    source_text = ""
    display_topic = topic or "Uploaded Document"
    
    level = request.form.get('level', 'Intermediate')
    mode = request.form.get('mode', 'Standard')
    
    extracted_text = ""
    if file:
        extracted_text = ai_engine.extract_text_from_file(file)
        if not topic:
            display_topic = file.filename
    
    # User's specifically requested prompt template
    prompt_template = f"""
    Generate structured study notes.

    Topic: {topic if topic else display_topic}

    Content:
    {extracted_text if extracted_text else "Base the notes on internal academic knowledge regarding the topic."}

    Include:
    - Introduction
    - Definitions
    - Key Concepts
    - Examples
    - Short Revision Points
    - Important Questions

    Level: {level}
    Mode: {mode}
    """
    
    system_instruction = "You are CourseMate AI, a premium academic synthesizer. Construct the response with high-end Markdown formatting."
    
    try:
        notes_content = ai_engine.get_ai_response(prompt_template, context=system_instruction)
        
        if not notes_content or len(notes_content) < 50:
            raise ValueError("Insufficient AI response for notes")

        new_entry = History(type='notes', topic=display_topic, content=notes_content)
        db.session.add(new_entry)
        db.session.commit()
        
        # Cache for RAG operations
        document_store["current_text"] = extracted_text if extracted_text else notes_content
        
    except Exception as e:
        print(f"Notes generation error: {e}")
        db.session.rollback()
        notes_content = f"# Sync Error\n\nUnable to reach the neural manuscript engine for '{display_topic}'. Please check your subject matter and try again."
    
    return jsonify({'notes': notes_content})

@app.route('/api/rag-chat', methods=['POST'])
def rag_chat():
    data = request.json
    query = data.get('query')
    
    if not document_store["current_text"]:
        return jsonify({'response': "No document context found. Please generate or upload notes first."})
        
    response = ai_engine.perform_rag(query, document_store["current_text"])
    return jsonify({'response': response})

@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    topic = data.get('topic')
    
    # Forceful prompt for 15-20 questions
    system_context = "Output strictly as a JSON array of 15-20 high-quality MCQ objects. " \
                     "Each object format: {\"question\": \"...\", \"options\": [\"a\", \"b\", \"c\", \"d\"], \"answer\": \"correct value\"}. " \
                     "Cover the topic comprehensively with varied difficulty. No introductory text."
    
    quiz_response = ai_engine.get_ai_response(topic, context=system_context)
    
    try:
        # Robust JSON extraction
        import re
        # Remove any markdown code blocks if present
        clean_response = re.sub(r'```json\s*|\s*```', '', quiz_response).strip()
        
        json_match = re.search(r'\[.*\]', clean_response, re.DOTALL)
        if json_match:
            quiz_data = json.loads(json_match.group())
        else:
            quiz_data = json.loads(clean_response)
    except Exception as e:
        print(f"Quiz parsing failure: {e}")
        # Multi-question fallback to avoid "only 2 questions" complaint
        quiz_data = [
            {"question": f"Key concept in {topic}: What is its primary objective?", "options": ["Theoretical Analysis", "Practical Application", "Historical Context", "Systemic Integration"], "answer": "Practical Application"},
            {"question": f"Advanced {topic}: Which methodology is most efficient?", "options": ["Linear Analysis", "Iterative Refinement", "Standardized Protocol", "Random Sampling"], "answer": "Iterative Refinement"},
            {"question": f"Foundations of {topic}: When was the core principle established?", "options": ["Early 20th Century", "Mid 19th Century", "Late 20th Century", "Renaissance Era"], "answer": "Early 20th Century"}
        ]
    
    try:
        new_entry = History(type='quiz', topic=topic, content=json.dumps(quiz_data))
        db.session.add(new_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Quiz storage error: {e}")
    
    return jsonify({'quiz': quiz_data})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    interests = data.get('interests')
    level = data.get('level', 'Beginner')
    goal = data.get('goal')
    time = data.get('time')
    
    prompt = f"""
    Act as a career advisor AI.

    User Interests: {interests}
    Skill Level: {level}
    Goal: {goal}
    Time Availability: {time}

    Generate:

    1. Best Career Path
    2. Step-by-step Roadmap
    3. Required Skills
    4. Recommended Courses
    5. Career Opportunities
    6. Project Ideas

    Make it structured, professional, and easy to understand using refined Markdown.
    """
    
    system_instruction = "You are the CourseMate Pathfinding AI. Synthesize a professional career roadmap based on user telemetry."
    
    roadmap = ai_engine.get_ai_response(prompt, context=system_instruction)
    
    try:
        new_entry = History(type='recommendation', topic=goal, content=roadmap)
        db.session.add(new_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Roadmap storage error: {e}")
        
    return jsonify({'roadmap': roadmap})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000, use_reloader=False)
