# database.py

import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            result TEXT,
            method TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(input_data, result, method):
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, input_data, result, method)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), input_data, result, method))
    conn.commit()
    conn.close()

def get_prediction_history():
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows
