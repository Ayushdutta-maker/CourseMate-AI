import os
import json
import google.generativeai as genai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import PyPDF2
import io

load_dotenv()

# Configure API keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Initialize NVIDIA Client (NIM handles OpenAI-compatible requests)
nvidia_client = None
if NVIDIA_KEY:
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_KEY
    )

class CourseMateAI:
    def __init__(self):
        # Predefined courses for recommendation logic
        self.courses = [
            "Introduction to Computer Science",
            "Advanced Python Programming",
            "Data Science for Beginners",
            "Machine Learning Specialization",
            "Web Development with HTML, CSS, and JS",
            "Digital Marketing Essentials",
            "Business Management & Leadership",
            "Quantum Physics for Enthusiasts",
            "Modern World History",
            "Calculus and Linear Algebra",
            "UX/UI Design Fundamentals",
            "Cybersecurity Ethics",
            "Financial Literacy & Investing"
        ]
        self.vectorizer = TfidfVectorizer()
        self.course_vectors = self.vectorizer.fit_transform(self.courses)

    def extract_text_from_file(self, file):
        """
        Extracts plain text from PDF or TXT files.
        """
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            try:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except Exception as e:
                return f"Error reading PDF: {str(e)}"
        elif filename.endswith('.txt'):
            try:
                return file.read().decode('utf-8')
            except Exception as e:
                return f"Error reading text file: {str(e)}"
        return ""

    def get_ai_response(self, prompt, context=""):
        """
        Generic AI response generator. 
        Multi-stage fallback: NVIDIA (NIM) -> Gemini -> Local Engine.
        """
        try:
            # 1. ATTEMPT NVIDIA NIM (Llama 3.1 8B is highly available)
            if nvidia_client:
                try:
                    completion = nvidia_client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct", 
                        messages=[
                            {"role": "system", "content": "You are CourseMate AI, a premium academic assistant. Provide high-density, structured academic info."},
                            {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
                        ],
                        temperature=0.6,
                        max_tokens=3000,
                        timeout=60.0 # Increased for larger batches (15-20 questions)
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    print(f"NVIDIA Signal Lost (Timeout or Error): {e}")

            # 2. FALLBACK TO GEMINI
            if GEMINI_KEY:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(
                        f"{context}\n\nUser: {prompt}",
                        generation_config=genai.types.GenerationConfig(max_output_tokens=3000)
                    )
                    return response.text
                except Exception as e:
                    print(f"Gemini Signal Lost: {e}")
            
            # 3. LOCAL SIMULATED INTELLIGENCE (Prevent 500 Errors)
            if "notes" in context.lower():
                return f"# Study Manuscript: {prompt}\n\n[SIMULATED OFFLINE MODE]\n\nThe academic engine is currently in local mode. {prompt} is an essential concept involving structured analysis and interdisciplinary application."
            
            return f"Synchronizing intelligence... Regarding '{prompt}', this is a core academic theme. (Local Mode Active)"

        except Exception as e:
            return f"Critical Engine Exception: {str(e)}"

    def recommend_courses(self, user_interests):
        """
        Basic ML recommendation logic using Sine Similarity.
        """
        user_vector = self.vectorizer.transform([user_interests])
        similarities = cosine_similarity(user_vector, self.course_vectors).flatten()
        related_indices = np.argsort(similarities)[-3:][::-1]
        
        # If no similarity (similarity sum is 0), return random/default
        if sum(similarities) == 0:
            return ["Intro to General Sciences", "Foundations of Learning", "Academic Excellence"]
            
        return [self.courses[i] for i in related_indices]

    def perform_rag(self, query, full_text):
        """
        Simple RAG implementation: Chunking -> Vector Search -> Contextual AI Response.
        """
        if not full_text or len(full_text) < 100:
            return self.get_ai_response(query, context="Note: No document context available.")

        # 1. Chunking
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
        
        # 2. Vector Search (Retreival)
        try:
            temp_vectorizer = TfidfVectorizer()
            chunk_vectors = temp_vectorizer.fit_transform(chunks)
            query_vector = temp_vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            relevant_context = "\n---\n".join([chunks[i] for i in top_indices if similarities[i] > 0.05])
            
            if not relevant_context:
                relevant_context = full_text[:2000] # Fallback to beginning of doc
        except:
            relevant_context = full_text[:2000]

        # 3. Augmentation & Generation
        system_instruction = "You are CourseMate RAG Engine. Use ONLY the provided context to answer. If not found, say so politely."
        return self.get_ai_response(query, context=f"{system_instruction}\n\nDOCUMENT CONTEXT:\n{relevant_context}")

# Singleton instance
ai_engine = CourseMateAI()
