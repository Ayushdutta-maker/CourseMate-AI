import urllib.request
import json

endpoints = [
    ('/', 'Your Personal AI Study Companion'),
    ('/dashboard', 'Your Academic Workspace'),
    ('/chat', 'AI Academic Assistant'),
    ('/notes', 'Study Notes Generator'),
    ('/quiz', 'Interactive Quiz Lab'),
    ('/recommendations', 'Career Pathfinder'),
    ('/history', 'Learning History Vault')
]

print('=== Live HTTP Verification against http://127.0.0.1:5000 ===')
all_passed = True
for path, expected in endpoints:
    url = f'http://127.0.0.1:5000{path}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            if expected in content:
                print(f'PASS: {path:18} -> HTTP {status} (Verified content: "{expected}")')
            else:
                print(f'FAIL: {path:18} -> Missing string "{expected}"')
                all_passed = False
    except Exception as e:
        print(f'FAIL: {path:18} -> Error: {e}')
        all_passed = False

print('\n=== Live API Verification ===')
try:
    with urllib.request.urlopen('http://127.0.0.1:5000/api/stats') as resp:
        stats = json.loads(resp.read().decode('utf-8'))
        print(f'PASS: /api/stats         -> HTTP 200 (Total Sessions: {stats["total_sessions"]})')
except Exception as e:
    print(f'FAIL: /api/stats -> Error: {e}')
    all_passed = False

try:
    with urllib.request.urlopen('http://127.0.0.1:5000/api/document-status') as resp:
        doc = json.loads(resp.read().decode('utf-8'))
        print(f'PASS: /api/document-status -> HTTP 200 (Active Doc: {doc["filename"]})')
except Exception as e:
    print(f'FAIL: /api/document-status -> Error: {e}')
    all_passed = False

if all_passed:
    print('\nALL LIVE ENDPOINTS AND PAGES ARE FULLY OPERATIONAL AND VERIFIED!')
else:
    print('\nSome tests failed.')
