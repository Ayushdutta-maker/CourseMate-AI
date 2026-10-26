import unittest
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db, History
from utils import ai_engine

class CourseMateTestSuite(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_01_public_landing_page(self):
        """Test public landing portal loads with editorial academic elements."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('CourseMate', content)
        self.assertIn('Your Personal AI Study Companion', content)
        self.assertIn('A Smarter Way to Learn', content)
        self.assertIn('Academic Course Recommendations', content)

    def test_02_dashboard_page(self):
        """Test application dashboard page loads with live historical statistics."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('Your Academic Workspace', content)
        self.assertIn('Total AI Sessions', content)
        self.assertIn('Recent Academic Footprint', content)

    def test_03_feature_pages_accessible(self):
        """Test that all primary educational routes render successfully."""
        routes = ['/chat', '/notes', '/quiz', '/recommendations', '/history']
        for route in routes:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)

    def test_04_api_stats(self):
        """Test live statistics endpoint reflects database records."""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('total_sessions', data)
        self.assertGreaterEqual(data['total_sessions'], 28)
        self.assertIn('notes_generated', data)
        self.assertIn('quizzes_completed', data)

    def test_05_course_vector_recommendation(self):
        """Test TF-IDF and cosine similarity course recommendation engine."""
        results = ai_engine.recommend_courses('machine learning deep neural networks python', top_k=3)
        self.assertEqual(len(results), 3)
        top_title = results[0]['title']
        self.assertIn('Machine Learning', top_title)
        self.assertTrue('match_pct' in results[0])

    def test_06_api_course_search(self):
        """Test course search API endpoint with TF-IDF cosine similarity."""
        response = self.client.post(
            '/api/course-search',
            data=json.dumps({'query': 'cybersecurity penetration testing network'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(len(data['results']) > 0)
        self.assertIn('Cybersecurity', data['results'][0]['title'])

    def test_07_api_chat(self):
        """Test conversational AI assistant API."""
        response = self.client.post(
            '/api/chat',
            data=json.dumps({'message': 'Explain binary search in 1 sentence.'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertTrue('response' in data)
        self.assertTrue(len(data['response']) > 15)

    def test_08_api_generate_quiz(self):
        """Test MCQ quiz generation API with validation and fallback."""
        response = self.client.post(
            '/api/generate-quiz',
            data=json.dumps({'topic': 'Data Structures', 'count': 5}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['topic'], 'Data Structures')
        self.assertGreaterEqual(len(data['quiz']), 3)
        first_q = data['quiz'][0]
        self.assertIn('question', first_q)
        self.assertIn('options', first_q)
        self.assertIn('answer', first_q)

    def test_09_api_history_records(self):
        """Test persistent history query endpoint."""
        response = self.client.get('/api/history?limit=10')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)

    def test_10_document_status_api(self):
        """Test RAG document context status endpoint."""
        response = self.client.get('/api/document-status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('has_document', data)
        self.assertIn('filename', data)

if __name__ == '__main__':
    unittest.main()
