import sqlite3
import os

db_path = 'instance/database.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='history';")
    row = cursor.fetchone()
    if row:
        print(row[0])
    else:
        print("Table 'history' not found.")
    conn.close()
else:
    print("Database file not found.")
